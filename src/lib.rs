use idek_basics::{
    idek::nalgebra::{Isometry3, Similarity3, Vector3},
    Array2D, ShapeBuilder,
};
use std::{collections::HashSet, f32::consts::FRAC_PI_2};
mod pcg;

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
    dirty: HashSet<(usize, usize)>,
}

impl Solver {
    pub fn new(tiles: Vec<Tile>, width: usize, height: usize) -> Self {
        let init_tile_set = vec![true; tiles.len()];
        let grid = Array2D::from_array(width, vec![init_tile_set; width * height]);

        /*
        let mut grid = grid;
        let tile = &mut grid[(3, 5)];
        tile.iter_mut().for_each(|b| *b = false);
        tile[0] = true;
        tile[3] = true;

        let mut rng = pcg::Rng::new();
        for tile in grid.data_mut() {
            tile.iter_mut().for_each(|b| {
                *b = rng.gen() & 1 == 0;
            })
        }
        */

        Self::from_grid(tiles, grid)
    }

    pub fn from_grid(tiles: Vec<Tile>, grid: Array2D<TileSet>) -> Self {
        Self {
            tiles,
            grid,
            dirty: HashSet::new(),
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
        let pos = self.dirty.iter().next().copied().unwrap();
        self.dirty.remove(&pos);

        // Create a new tile set
        let mut new_tile_set = self.grid[pos].clone();
        let neighborhood = sample_bndchk(&self.grid, pos);

        // Determine for each neighbor...
        for (neigh_idx, neigh) in neighborhood.into_iter().enumerate() {
            if let Some(neigh) = neigh {
                // ... For each possible tile that neighbor could be...
                for (idx, cond) in self.grid[neigh].iter().enumerate() {
                    if *cond {
                        // ... Whether or not that tile set contains us
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
                    self.dirty.insert(neigh);
                }
            }
            self.grid[pos] = new_tile_set;
        }

        ControlFlow::Continue
    }

    fn step_random(&mut self) -> ControlFlow {
        // if finished collapsing, return finished else continue
        todo!("Shannon entropy random selection")
    }
}

fn sample_bndchk<T>(arr: &Array2D<T>, (x, y): (usize, usize)) -> [Option<(usize, usize)>; 4] {
    [(-1, 0), (0, -1), (1, 0), (0, 1)]
        .map(|(dx, dy)| bnd_chk(x, dx, arr.width()).zip(bnd_chk(y, dy, arr.height())))
}

fn bnd_chk(x: usize, dx: isize, max: usize) -> Option<usize> {
    if dx < 0 {
        x.checked_sub(-dx as usize)
    } else {
        (x < max).then(|| x + dx as usize)
    }
}
