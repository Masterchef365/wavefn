use idek_basics::{
    idek::nalgebra::{Isometry3, Similarity3, Vector3},
    Array2D, ShapeBuilder,
};
use pcg::Rng;
use std::{collections::HashSet, f32::consts::FRAC_PI_2};
pub mod pcg;

pub type Coord = (usize, usize);

pub type Grid = Array2D<TileSet>;

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
    //[b, c, d, a]
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

#[derive(Clone)]
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
    grid: Grid,
    /// Tile coordinates to be updated next, if any
    dirty: Vec<Coord>,
}

pub fn init_grid(width: usize, height: usize, tiles: &[Tile]) -> Grid {
    let init_tile_set = vec![true; tiles.len()];
    let data = vec![init_tile_set; width * height];
    Array2D::from_array(width, data)
}

impl Solver {
    pub fn from_grid(tiles: Vec<Tile>, grid: Grid) -> Self {
        Self {
            tiles,
            grid,
            dirty: Vec::new(),
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    pub fn dirty(&self) -> &[Coord] {
        &self.dirty
    }

    pub fn step(&mut self, rng: &mut Rng) -> ControlFlow {
        loop {
            if self.dirty.is_empty() {
                break self.step_random(rng);
            } else {
                self.step_dirty();
            }
        }
    }

    fn step_dirty_part(&mut self) -> ControlFlow {
        let mut lowest = None;
        let mut lowest_entropy = usize::MAX;
        for (idx, pos) in self.dirty.iter().enumerate() {
            let entropy = tile_entropy(&self.grid, *pos);
            if entropy < lowest_entropy {
                lowest = Some(idx);
                lowest_entropy = entropy;
            }
        }

        let pos = match lowest {
            Some(idx) => self.dirty.remove(idx),
            None => return ControlFlow::Continue,
        };

        // Create a new tile set
        let new_tile_set = update_tile(&self.grid, &self.tiles, pos);

        // Early exit if contradiction
        if new_tile_set.iter().all(|f| !f) {
            return ControlFlow::Contradiction;
        }

        if new_tile_set != self.grid[pos] {
            // ... and if so, mark us neighbors as dirty:
            self.dirty.push(pos);
            self.set_neighbors_dirty(pos);
            self.grid[pos] = new_tile_set;
        }

        ControlFlow::Continue

    }

    fn set_neighbors_dirty(&mut self, pos: Coord) {
        for neigh in neighbor_coords(&self.grid, pos) {
            if let Some(neigh) = neigh {
                if count_tileset(&self.grid[neigh]) != 1 {
                    self.dirty.push(neigh);
                }
            }
        }
    }

    fn step_dirty(&mut self) -> ControlFlow {
        /*
           let mut ctrl = self.step_dirty_part();
           if ctrl == ControlFlow::Contradiction {
           ctrl = self.step_dirty_part();
           }*/
        self.step_dirty_part()
    }

    fn step_random(&mut self, rng: &mut Rng) -> ControlFlow {
        //todo!("Shannon entropy random selection")
        // Find lowest entropy
        // Find lowest entropy
        let mut lowest_n = usize::MAX;
        for y in 0..self.grid.height() {
            for x in 0..self.grid.width() {
                if count_tileset(&self.grid[(x, y)]) > 1 {
                    let n = tile_entropy(&self.grid, (x, y));
                    if n < lowest_n {
                        lowest_n = n;
                    }
                }
            }
        }

        // Find all tiles with that entropy
        let mut lowest = vec![];

        for y in 0..self.grid.height() {
            for x in 0..self.grid.width() {
                if count_tileset(&self.grid[(x, y)]) > 1 {
                    let n = tile_entropy(&self.grid, (x, y));
                    if n == lowest_n {
                        lowest.push((x, y));
                    }
                }
            }
        }

        // Randomly select one and collapse it, marking its neighbors dirty
        if let Some(&pos) = choose(rng, &lowest) {
            let ones = self.grid[pos]
                .iter()
                .enumerate()
                .filter_map(|(i, p)| p.then(|| i))
                .collect::<Vec<_>>();
            let idx = ones[rng.gen() as usize % ones.len()];
            self.grid[pos][idx] = false;

            self.set_neighbors_dirty(pos);

            self.dirty.push(pos);
            ControlFlow::Continue
        } else {
            ControlFlow::Finish
        }
    }
}

pub fn update_tile(grid: &Grid, tiles: &[Tile], pos: Coord) -> TileSet {
    let mut set = grid[pos].clone();
    let neighborhood = neighbor_coords(&grid, pos);

    // For each tile this cell could be...
    for idx in 0..tiles.len() {
        if !set[idx] {
            continue;
        }

        let mut present = true;

        // Check for all sides whether it's possible to be this tile
        for (side, neigh_pos) in neighborhood.into_iter().enumerate() {
            if let Some(neigh) = neigh_pos.map(|p| &grid[p]) {
                present &= tiles[idx].rules[side]
                    .iter()
                    .zip(neigh)
                    .any(|(t, n)| t & n);
            }
        }

        set[idx] = present;
    }

    set
}

fn tile_entropy(grid: &Grid, pos: Coord) -> usize {
    let mut n = count_tileset(&grid[pos]);

    for neigh in neighbor_coords(grid, pos) {
        if let Some(neigh) = neigh {
            n += count_tileset(&grid[neigh]);
        } else {
            n += grid[pos].len();
        }
    }
    n
}

pub fn count_tileset(ts: &TileSet) -> usize {
    //ts.iter().filter(|p| **p).count()
    ts.iter().map(|t| *t as usize).sum()
}

fn choose<'a, T>(rng: &mut Rng, arr: &'a [T]) -> Option<&'a T> {
    if arr.is_empty() {
        None
    } else {
        let idx = rng.gen() as usize % arr.len();
        Some(&arr[idx])
    }
}

fn neighbor_coords<T>(arr: &Array2D<T>, (x, y): Coord) -> [Option<Coord>; 4] {
    [(1, 0), (0, 1), (-1, 0), (0, -1)]
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
