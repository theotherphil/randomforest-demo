
extern crate image;
extern crate imageproc;
extern crate rand;
extern crate randomforest;

use std::path::Path;
use std::collections::HashMap;
use std::collections::HashSet;
use rand::{Rng, thread_rng, ThreadRng};

use std::f64;
use image::{Rgb, RgbImage};
use imageproc::utils::load_image_or_panic;
use imageproc::definitions::HasWhite;
use imageproc::drawing::draw_cross_mut;
use imageproc::regionlabelling::{connected_components, Connectivity};
use randomforest::*;
use randomforest::stump::*;
use randomforest::hyperplane::*;

/// Labelled data. labels and data have each equal length and
/// labels[i] is the label for data[i].
struct Labelled<L, D> {
    labels: Vec<L>,
    data: Vec<D>
}

/// All non-white connected components are treated as data points. Returns the colour
/// and centre of mass of each component.
fn create_labelled_data(image: &RgbImage) -> Labelled<Rgb<u8>, (f64, f64)> {
    let ccs = connected_components(image, Connectivity::Eight, Rgb::white());
    let mut elements: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let (width, height) = image.dimensions();

    for y in 0..height {
        for x in 0..width {
            let component = ccs.get_pixel(x, y)[0];
            let cc = elements.entry(component).or_insert(Vec::<(u32, u32)>::new());
            cc.push((x, y));
        }
    }

    let mut centres = vec![];
    let mut colours = vec![];

    for (component, members) in &elements {
        // Ignore background component
        if *component == 0u32 {
            continue;
        }

        let sum = members
            .iter()
            .fold((0f64, 0f64), |acc, p| {
                (acc.0 + p.0 as f64, acc.1 + p.1 as f64)
            });

        let count = members.len() as f64;
        centres.push((sum.0 / count, sum.1 / count));

        // All elements of the connected component have the same colour.
        // Read it from the first element.
        let colour = image.get_pixel(members[0].0, members[0].1);
        colours.push(*colour);
    }

    Labelled {
        labels: colours,
        data: centres
    }
}

fn draw_labelled_data(data: &Labelled<Rgb<u8>, (f64, f64)>, width: u32, height: u32)-> RgbImage {
    let mut crosses = RgbImage::from_raw(width, height, vec![255; (width * height * 3) as usize]).unwrap();

    for i in 0..data.labels.len() {
        draw_cross_mut(&mut crosses, data.labels[i], data.data[i].0 as i32, data.data[i].1 as i32);
    }

    crosses
}

struct Transformer {
    // Uses to scale image input data to [0, 1]
    scale_factor: f64,
    // Maps image colour to index
    colour_to_index: HashMap<Rgb<u8>, usize>,
    // Maps index to image colour
    index_to_colour: HashMap<usize, Rgb<u8>>
}

impl Transformer {
    fn create(width: u32, height: u32, data: &Labelled<Rgb<u8>, (f64, f64)>) -> Transformer {
        let mut colour_to_index = HashMap::new();
        let mut index_to_colour = HashMap::new();

        for &l in data.labels.iter() {
            if colour_to_index.contains_key(&l) {
                continue;
            }
            let index = colour_to_index.len();
            colour_to_index.insert(l, index);
            index_to_colour.insert(index, l);
        }

        Transformer {
            scale_factor: 1.0 / f64::max(width as f64, height as f64),
            colour_to_index: colour_to_index,
            index_to_colour: index_to_colour
        }
    }
}

fn create_dataset(trans: &Transformer, input: &Labelled<Rgb<u8>, (f64, f64)>) -> Dataset {
    let mut labels = vec![];
    let mut data = vec![];

    for i in 0..input.labels.len() {
        let l = input.labels[i];
        let d = input.data[i];

        let idx = trans.colour_to_index[&l];
        let scaled = vec![d.0 * trans.scale_factor, d.1 * trans.scale_factor];

        labels.push(idx);
        data.push(scaled);
    }

    Dataset {
        labels: labels,
        data: data
    }
}

fn train_stump_forest(trans: &Transformer,
                params: ForestParameters,
                input: &Labelled<Rgb<u8>, (f64, f64)>) -> Forest<Stump> {
    let data = create_dataset(trans, input);

    let mut generator = StumpGenerator {
        rng: thread_rng(),
        num_dims: data.data[0].len(),
        min_thresh: 0f64,
        max_thresh: 1f64
    };

    Forest::train(params, &mut generator, &data)
}

fn train_plane_forest(trans: &Transformer,
                params: ForestParameters,
                input: &Labelled<Rgb<u8>, (f64, f64)>) -> Forest<Plane> {
    let data = create_dataset(trans, input);

    let mut generator = PlaneGenerator {
        rng: thread_rng(),
        num_dims: data.data[0].len(),
        min_thresh: 0f64,
        max_thresh: 1f64
    };

    Forest::train(params, &mut generator, &data)
}

fn blend(dist: &Distribution, trans: &Transformer) -> Rgb<u8> {
    let mut pix = vec![0f64, 0f64, 0f64];
    for i in 0..dist.probs.len() {
        let prob = dist.probs[i];
        let colour = trans.index_to_colour[&i];

        pix[0] += colour.data[0] as f64 * prob;
        pix[1] += colour.data[1] as f64 * prob;
        pix[2] += colour.data[2] as f64 * prob;
    }
    Rgb([pix[0] as u8, pix[1] as u8, pix[2] as u8])
}

// leaf distributions aren't weighted by number of training samples
// of a given class. should they be?

fn create_forest_parameters(labelled: &Labelled<Rgb<u8>, (f64, f64)>) -> ForestParameters {
    ForestParameters {
        num_trees: 200usize,
        depth: 9usize,
        num_classes: labelled
            .labels
            .iter()
            .fold(HashSet::new(), |mut acc, l| { acc.insert(l); acc })
            .len(),
        num_candidates: 500usize
    }
}

fn main() {
    //let source_path = Path::new("./src/four-class-spiral.png");
    let source_path = Path::new("./src/cazzo.png");
    let centres_path = Path::new("./src/centres.png");
    let classified_path = Path::new("./src/classification.png");
    let confidence_path = Path::new("./src/confidence.png");

    let image = load_image_or_panic(&source_path).to_rgb();
    let labelled = create_labelled_data(&image);
    let crosses = draw_labelled_data(&labelled, image.width(), image.height());
    let _ = crosses.save(centres_path);

    // Transforms between image coordinates/colours and [0, 1]/indices
    let trans = Transformer::create(image.width(), image.height(), &labelled);

    let params = create_forest_parameters(&labelled);
    //let forest = train_stump_forest(&trans, params, &labelled);
    let forest = train_plane_forest(&trans, params, &labelled);

    println!("trained forest");

    let mut classified = RgbImage::new(image.width(), image.height());
    let mut confidence = RgbImage::new(image.width(), image.height());

    for y in 0..image.height() {
        for x in 0..image.width() {
            let p = vec![x as f64 * trans.scale_factor, y as f64 * trans.scale_factor];
            let dist = forest.classify(&p);
            classified.put_pixel(x, y, blend(&dist, &trans));

            let entropy = dist
                .probs
                .iter()
                .fold(0f64, |acc, p| if *p > 0f64 { acc - p * p.log2() } else { acc });

            let level = (255f64 * entropy / (params.num_classes as f64).log2()) as u8;
            confidence.put_pixel(x, y, Rgb([level, level, level]))
        }
    }

    let _ = classified.save(classified_path);
    let _ = confidence.save(confidence_path);
}
