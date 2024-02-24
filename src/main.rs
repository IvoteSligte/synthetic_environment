use grid::*;
use glam::IVec2;
use noise::NoiseFn;

use tuple_traits::*;

mod tuple_traits {
    pub trait AsTuple {
        type Output;

        fn as_tuple(self) -> Self::Output;
    }

    impl AsTuple for glam::IVec2 {
        type Output = (usize, usize);
        
        fn as_tuple(self) -> Self::Output {
            debug_assert!(self.x >= 0 && self.y >= 0);
            (self.x as usize, self.y as usize)
        }
    }

    pub trait FromTuple {
        type Input;

        fn from_tuple(tuple: Self::Input) -> Self;
    }

    impl FromTuple for glam::IVec2 {
        type Input = (usize, usize);

        fn from_tuple(tuple: Self::Input) -> Self {
            let (x, y) = tuple;
            debug_assert!(x <= i32::MAX as usize && y <= i32::MAX as usize);
            Self::new(x as i32, y as i32)
        }
    }
}

const SIZE: usize = 512;

/// learning rate of grid, range (0, 1]
const LEARNING_RATE: f64 = 0.5;

const MAP_SCALE: f64 = 100.0;

const KERNEL: &[(IVec2, f64)] = &[
    (IVec2::new(-2, -2), 0.003663),
    (IVec2::new(-1, -2), 0.014652),
    (IVec2::new(0, -2), 0.025641),
    (IVec2::new(1, -2), 0.014652),
    (IVec2::new(2, -2), 0.003663),
    (IVec2::new(-2, -1), 0.014652),
    (IVec2::new(-1, -1), 0.0586081),
    (IVec2::new(0, -1), 0.0952381),
    (IVec2::new(1, -1), 0.0586081),
    (IVec2::new(2, -1), 0.014652),
    (IVec2::new(-2, 0), 0.025641),
    (IVec2::new(-1, 0), 0.0952381),
    (IVec2::new(0, 0), 0.150183),
    (IVec2::new(1, 0), 0.0952381),
    (IVec2::new(2, 0), 0.025641),
    (IVec2::new(-2, 1), 0.014652),
    (IVec2::new(-1, 1), 0.0586081),
    (IVec2::new(0, 1), 0.0952381),
    (IVec2::new(1, 1), 0.0586081),
    (IVec2::new(2, 1), 0.014652),
    (IVec2::new(-2, 2), 0.003663),
    (IVec2::new(-1, 2), 0.014652),
    (IVec2::new(0, 2), 0.025641),
    (IVec2::new(1, 2), 0.014652),
    (IVec2::new(2, 2), 0.003663),
];

const FPS: f64 = 2.0;

fn next_pixel<F>(pixel: (usize, usize), src: &Grid<f64>, map_fn: F) -> f64 
where F: Fn(f64, f64) -> f64{
    let original = src[pixel];
    let mut target = 0.0;
    let pixel = IVec2::from_tuple(pixel);
    let grid_size = IVec2::new(src.cols() as i32, src.rows() as i32);

    for &(offset, weight) in KERNEL {
        target += map_fn(original, src[(pixel + offset).rem_euclid(grid_size).as_tuple()]) * weight;
    }
    original + LEARNING_RATE * target
}

fn random_square_grid<const N: usize>(seed: u32, noise_scale: f64) -> Grid<f64> {
    let noise = noise::Perlin::new(seed);
    let mut grid = Grid::init(N, N, 0.0);

    for x in 0..N {
        let xf = x as f64 / N as f64 * noise_scale;
        for y in 0..N {
            let yf = y as f64 / N as f64 * noise_scale;
            grid[(x, y)] = noise.get([xf, yf]) * 0.5 + 0.5;
        }
    }
    grid
}

fn image_from_grid(grid: &Grid<f64>) -> show_image::Image {
    let image_data = grid.iter_rows().flatten().map(|value| (value * 256.0) as u8).collect::<Vec<u8>>().into_boxed_slice();
    let image = show_image::BoxImage::new(show_image::ImageInfo::mono8(grid.rows() as u32, grid.cols() as u32), image_data);

    show_image::Image::Box(image)
}

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut random_grid = random_square_grid::<SIZE>(10, 40.0);
    let mut empty_grid = Grid::init(SIZE, SIZE, 0.0f64);
    let noise = noise::Perlin::new(0);
    let map_fn = move |org, comp| noise.get([org * MAP_SCALE, comp * MAP_SCALE]);
    let window = show_image::create_window("image", Default::default()).unwrap();
    let mut last_frame = std::time::Instant::now();

    let mut i = 0;
    loop {
        let (src_grid, dst_grid) = if i % 2 == 0 {
            (&random_grid, &mut empty_grid)
        } else {
            (&empty_grid, &mut random_grid)
        };
        for (pixel, _) in src_grid.indexed_iter() {
            dst_grid[pixel] = next_pixel(pixel, src_grid, map_fn);
        }
        i += 1;
        // TODO: frame num
        window.set_image("image-frame_num", image_from_grid(dst_grid)).unwrap();
        //std::thread::sleep(std::time::Duration::from_secs_f64(1.0 / FPS).checked_sub(last_frame.elapsed()).unwrap_or_default());
        last_frame = std::time::Instant::now();

        println!("Average energy: {}", dst_grid.iter().sum::<f64>() / (SIZE * SIZE) as f64);
    }
}

