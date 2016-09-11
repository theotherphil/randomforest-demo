
extern crate image;
extern crate imageproc;

use std::path::Path;
use std::collections::HashMap;

use image::{Rgb, RgbImage};
use imageproc::utils::load_image_or_panic;
use imageproc::definitions::HasWhite;
use imageproc::drawing::draw_cross_mut;
use imageproc::regionlabelling::{connected_components, Connectivity};

/// Labelled data. labels and data have each equal and
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

fn main() {
    let source_path = Path::new("./src/grid.png");
    let target_path = Path::new("./src/centres.png");

    let image = load_image_or_panic(&source_path).to_rgb();
    let labelled = create_labelled_data(&image);
    let crosses = draw_labelled_data(&labelled, image.width(), image.height());
    let _ = crosses.save(target_path);
}
