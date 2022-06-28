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

use std::collections::HashSet;

use idek_basics::{Array2D, GraphicsBuilder};

/*
enum Symmetry {
    /// No transformation
    Identity,
    /// One 45-degree tf
    Rot2,
    /// Rotate 4 ways 45 degrees in the plane
    Rot4,
    /// Mirror _over_ the y axis (inverts art indices)
    MirrorX,
    /// Mirror _over_ the x axis (inverts art indices)
    MirrorY,
}

struct Shape {
    art: GraphicsBuilder,
    sym: Symmetry,
    /// Each number corresponds to an interface type. If left at 0, any interface can connect.
    conn: [u32; 4],
    /// Weight relative to other shapes
    weight: f32,
}

/// Convert a set of shapes into a set of tiles useable by the solver as rules
fn compile_tiles(shapes: &[Shape]) -> (Vec<Tile>, Vec<f32>) {
    let weights = todo!();
    let tiles = todo!();
    (tiles, weights)
}
*/

fn draw_solve(solver: &Solver) -> GraphicsBuilder {
    let mut gb = GraphicsBuilder::new();
}

fn extend_gb(dest: &mut GraphicsBuilder, src: &GraphicsBuilder, ) {
    let base = dest.vertices.len() as u32;
}

/// Each entry refers to one other tile in an array of Tiles (a RuleSet)
/// The boolean is true if the tile can be made adjacent to it
type TileSet = Vec<bool>;

struct Tile {
    /// Transformed art
    art: GraphicsBuilder,
    /// Sets of tiles which can be adjacent to this one
    rules: [TileSet; 4],
}

enum ControlFlow {
    Contradiction,
    Finish,
    Continue,
    //ContinuePropagate,
    //ContinueChooseNew,
}

struct Solver {
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

        Self::from_grid(tiles, grid)
    }

    pub fn from_grid(tiles: Vec<Tile>, grid: Array2D<TileSet>) -> Self {
        Self {
            tiles,
            grid,
            dirty: HashSet::new(),
        }
    }

    pub fn step(&mut self) -> ControlFlow {
        if self.dirty.is_empty() {
            self.step_dirty()
        } else {
            self.step_random()
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
