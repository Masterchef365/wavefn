use idek_basics::{
    idek::nalgebra::{Isometry3, Similarity3, Vector3},
    Array2D, ShapeBuilder,
};
use pcg::Rng;
use std::{collections::HashSet, f32::consts::FRAC_PI_2};
pub mod pcg;

/*
Connection directions are clockwise, like so:
      ^
      3
      |
<2 -- + -- 0>
      |
      1
      V

+--> +X
|
V
+Y
*/

pub enum Symmetry {
    // /// No transformation
    // Identity,
    /// One 45-degree tf
    Rot2,
    /// Rotate 4 ways 45 degrees in the plane
    Rot4,
    // /// Mirror _over_ the y axis (inverts art indices)
    // MirrorX,
    // /// Mirror _over_ the x axis (inverts art indices)
    // MirrorY,
}

#[derive(Clone)]
pub struct Shape {
    pub art: ShapeBuilder,
    /// Each number corresponds to an interface type.
    pub conn: [u32; 4],
    // /// Weight relative to other shapes
    // pub weight: f32,
}

pub fn apply_symmetry(shape: &Shape, sym: Symmetry) -> Vec<Shape> {
    match sym {
        Symmetry::Rot2 => {
            vec![
                shape.clone(),
                Shape {
                    art: rot_shapeb(&shape.art, FRAC_PI_2),
                    conn: rot_conn_cw_90(shape.conn),
                },
            ]
        }
        Symmetry::Rot4 => {
            vec![
                shape.clone(),
                Shape {
                    art: rot_shapeb(&shape.art, FRAC_PI_2),
                    conn: rot_conn_cw_90(shape.conn),
                },
                Shape {
                    art: rot_shapeb(&shape.art, FRAC_PI_2 * 2.),
                    conn: rot_conn_cw_90(rot_conn_cw_90(shape.conn)),
                },
                Shape {
                    art: rot_shapeb(&shape.art, FRAC_PI_2 * 3.),
                    conn: rot_conn_cw_90(rot_conn_cw_90(rot_conn_cw_90(shape.conn))),
                },
            ]
        }
    }
}

fn rot_conn_cw_90([a, b, c, d]: [u32; 4]) -> [u32; 4] {
    [d, a, b, c]
}

pub fn rot_shapeb(art: &ShapeBuilder, angle: f32) -> ShapeBuilder {
    let mut s = ShapeBuilder::new();
    let tf = Similarity3::from_isometry(Isometry3::translation(0.5, 0.5, 0.), 1.)
        * Similarity3::from_isometry(Isometry3::rotation(Vector3::new(0., 0., angle)), 1.)
        * Similarity3::from_isometry(Isometry3::translation(-0.5, -0.5, 0.), 1.);

    s.push_tf(tf);
    s.append(art);
    s
}

fn invert_indices(i: &[u32]) -> Vec<u32> {
    i.chunks_exact(3)
        .map(|c| [c[2], c[1], c[0]])
        .flatten()
        .collect()
}

/// Convert a set of shapes into a set of tiles useable by the solver as rules
//pub fn compile_tiles(shapes: &[Shape]) -> (Vec<Tile>, Vec<f32>) {
pub fn compile_tiles(shapes: &[Shape]) -> Vec<Tile> {
    let mut tiles = vec![];

    for shape in shapes {
        let art = shape.art.clone();

        // Check our connection points against possible partners
        let pairs = [(0, 2), (1, 3), (2, 0), (3, 1)];
        let rules: [TileSet; 4] = pairs.map(|(side, partner_side)| {
            shapes
                .iter()
                .map(|partner| shape.conn[side] == partner.conn[partner_side])
                .collect()
        });

        tiles.push(Tile { art, rules });
    }

    tiles
}

/// Each entry refers to one other tile in an array of Tiles (a RuleSet)
/// The boolean is true if the tile can be made adjacent to it
/// TODO: Use a more efficient data structure!
///     * densely packed bits, u8s or maybe u128s or SIMD?
///     * Sorted array of usize? (indices)
pub type TileSet = Vec<bool>;

pub struct Tile {
    /// Transformed art
    pub art: ShapeBuilder,
    /// Sets of tiles which can be adjacent to this one
    pub rules: [TileSet; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFlow {
    Contradiction,
    Finish,
    Continue,
    //ContinuePropagate,
    //ContinueChooseNew,
}

pub struct Solver {
    /// Tiles in use
    tiles: Vec<Tile>,
    /// Grid of tile sets. Invalid if all false.
    grid: Array2D<TileSet>,
    /// Tile coordinates to be updated next, if any
    dirty: Vec<(usize, usize)>,
    /// Random number generator
    rng: Rng,
}

pub fn init_grid(width: usize, height: usize, tiles: &[Tile]) -> Array2D<TileSet> {
    let init_tile_set = vec![true; tiles.len()];
    let data = vec![init_tile_set; width * height];
    Array2D::from_array(width, data)
}

impl Solver {
    pub fn from_grid(tiles: Vec<Tile>, grid: Array2D<TileSet>) -> Self {
        Self {
            rng: Rng::new(),
            tiles,
            grid,
            dirty: Vec::new(),
        }
    }

    pub fn grid(&self) -> &Array2D<TileSet> {
        &self.grid
    }

    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    pub fn step(&mut self) -> ControlFlow {
        if self.dirty.is_empty() {
            self.step_random()
        } else {
            self.step_dirty()
        }
    }

    fn step_dirty(&mut self) -> ControlFlow {
        // Determine which tile set to update
        // Unwrap is okay because we are only called from step(); array length is checked there
        let idx = self.rng.gen() as usize % self.dirty.len();
        let pos = self.dirty.remove(idx);

        // Create a new tile set
        let mut new_tile_set = self.grid[pos].clone();
        let neighborhood = neighbor_coords(&self.grid, pos);

        // Determine for each neighbor...
        for (neigh_idx, neigh) in neighborhood.into_iter().enumerate() {
            if let Some(neigh) = neigh {
                // ... For each possible tile that neighbor could be...
                for (idx, cond) in self.grid[neigh].iter().enumerate() {
                    if *cond {
                        // ... Whether or not that tile intersects with our set
                        new_tile_set[idx] &= self.tiles[idx].rules[neigh_idx][idx];
                    }
                }
            }
        }

        // Early exit if contradiction
        if new_tile_set.iter().all(|f| !f) {
            return ControlFlow::Contradiction;
        }

        // Determine if we made a change...
        if new_tile_set != self.grid[pos] {
            // ... and if so, mark neighbors as dirty:
            for neigh in neighborhood {
                if let Some(neigh) = neigh {
                    dbg!(neigh);
                    self.dirty.push(neigh);
                }
            }
            self.grid[pos] = new_tile_set;
        }

        ControlFlow::Continue
    }

    fn step_random(&mut self) -> ControlFlow {
        //todo!("Shannon entropy random selection")
        let lowest_n = self
            .grid()
            .data()
            .iter()
            .map(count_tileset)
            .filter(|&c| c > 1)
            .min()
            .expect("No tiles");
        let mut lowest = vec![];

        for y in 0..self.grid.height() {
            for x in 0..self.grid.width() {
                let n = count_tileset(&self.grid[(x, y)]);
                if n == lowest_n {
                    lowest.push((x, y));
                }
            }
        }

        if let Some(lowest) = choose(&mut self.rng, &lowest) {
            self.dirty.push(*lowest);
            ControlFlow::Continue
        } else {
            ControlFlow::Finish
        }
    }
}

fn count_tileset(ts: &TileSet) -> usize {
    ts.iter().filter(|p| **p).count()
}

fn choose<'a, T>(rng: &mut Rng, arr: &'a [T]) -> Option<&'a T> {
    if arr.is_empty() {
        None
    } else {
        let idx = rng.gen() as usize % arr.len();
        Some(&arr[idx])
    }
}

fn neighbor_coords<T>(arr: &Array2D<T>, (x, y): (usize, usize)) -> [Option<(usize, usize)>; 4] {
    [(-1, 0), (0, -1), (1, 0), (0, 1)]
        .map(|(dx, dy)| bnd_chk(x, dx, arr.width()).zip(bnd_chk(y, dy, arr.height())))
}

fn bnd_chk(x: usize, dx: isize, max: usize) -> Option<usize> {
    if dx < 0 {
        x.checked_sub(-dx as usize)
    } else {
        let x = x + dx as usize;
        (x < max).then(|| x)
    }
}
