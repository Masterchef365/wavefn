use std::collections::HashSet;

use idek::{prelude::*, IndexBuffer};
use idek_basics::{
    idek::{
        self,
        nalgebra::{Isometry2, Isometry3, Matrix4, Similarity3},
    },
    Array2D, ShapeBuilder,
};
use wavefn::{
    apply_symmetry, compile_tiles, count_tileset, init_grid, pcg::Rng, ControlFlow, Grid, Shape,
    Solver, Symmetry, Tile, TileSet,
};

fn main() -> Result<()> {
    launch::<_, CubeDemo>(Settings::default().vr_if_any_args())
}

struct CubeDemo {
    line_verts: VertexBuffer,
    line_indices: IndexBuffer,
    line_gb: ShapeBuilder,
    line_shader: Shader,

    tri_verts: VertexBuffer,
    tri_indices: IndexBuffer,
    tri_gb: ShapeBuilder,

    grid: Grid,
    solver: Solver,

    control: ControlFlow,

    rng: Rng,

    frame: usize,
}

const CONN_WALL: u32 = 0;
const CONN_PATH: u32 = 1;

impl App for CubeDemo {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let mut line_gb = ShapeBuilder::new();
        let mut tri_gb = ShapeBuilder::new();

        let mut shapes = vec![];

        // Right turn
        shapes.extend(apply_symmetry(
            &Shape {
                art: cons_shape(path_right),
                conn: [CONN_PATH, CONN_PATH, CONN_WALL, CONN_WALL],
                //weight: 1.,
            },
            Symmetry::Rot4,
        ));

        // 3-way path
        shapes.extend(apply_symmetry(
            &Shape {
                art: cons_shape(path_3way),
                conn: [CONN_PATH, CONN_PATH, CONN_WALL, CONN_PATH],
                //weight: 1.,
            },
            Symmetry::Rot4,
        ));

        /*
        // End cap
        shapes.extend(apply_symmetry(
            &Shape {
                art: cons_shape(path_cap),
                conn: [CONN_WALL, CONN_PATH, CONN_WALL, CONN_WALL],
                //weight: 1.,
            },
            Symmetry::Rot4,
        ));

        // 4-way path
        shapes.push(Shape {
            art: cons_shape(path_4way),
            conn: [CONN_PATH; 4],
            //weight: 1.,
        });
        */

        // Straight path
        shapes.extend(apply_symmetry(
            &Shape {
                art: cons_shape(path_straight),
                conn: [CONN_WALL, CONN_PATH, CONN_WALL, CONN_PATH],
                //weight: 1.,
            },
            Symmetry::Rot2,
        ));

        /*
        for shape in &shapes {
            println!("{:?}", shape.conn);
        }
        */

        let tiles = compile_tiles(&shapes);

        /*
        for (idx, tile) in tiles.iter().enumerate() {
            println!("   {}:", idx);
            for (set_idx, set) in tile.rules.iter().enumerate() {
                println!("    {}: {:?}", set_idx, set);
            }
            println!();
        }
        */

        let mut rng = Rng::new();

        let grid = new_grid(&mut rng, &tiles);

        let solver = Solver::from_grid(tiles.clone(), grid.clone());

        draw_tile_grid(
            &mut line_gb,
            &solver.grid(),
            &solver.tiles(),
            Default::default(),
        );
        draw_tile_grid(
            &mut line_gb,
            &solver.grid(),
            &solver.tiles(),
            Default::default(),
        );

        path_right(&mut tri_gb);

        let line_verts = ctx.vertices(&line_gb.vertices, true)?;
        let line_indices = ctx.indices(&line_gb.indices, true)?;

        let tri_verts = ctx.vertices(&tri_gb.vertices, true)?;
        let tri_indices = ctx.indices(&tri_gb.indices, true)?;

        let line_shader = ctx.shader(
            DEFAULT_VERTEX_SHADER,
            DEFAULT_FRAGMENT_SHADER,
            Primitive::Lines,
        )?;

        Ok(Self {
            line_verts,
            line_indices,
            line_gb,
            line_shader,

            tri_verts,
            tri_indices,
            tri_gb,

            rng,

            control: ControlFlow::Continue,

            grid,

            solver,

            frame: 0,
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        //let frame = self.frame % (90*1) == 0;
        let frame = self.frame % 1 == 0;
        //let frame = true;
        let cont = self.control == ControlFlow::Continue;

        if frame && cont {
            //for _ in 0..300 {
            self.control = self.solver.step(&mut self.rng);
            if self.control == ControlFlow::Contradiction {
                dbg!(self.control);
                //self.solver = Solver::from_grid(self.solver.tiles().to_vec(), self.grid.clone());
                //self.control = ControlFlow::Continue;
            }
            /*if self.control == ControlFlow::Finish {
                    break;
                }
            }*/

            self.line_gb.clear();

            draw_solver(&mut self.line_gb, &self.solver);

            assert!(!self.line_gb.vertices.is_empty());
            assert!(!self.line_gb.indices.is_empty());
            ctx.update_vertices(self.line_verts, &self.line_gb.vertices)?;
            ctx.update_indices(self.line_indices, &self.line_gb.indices)?;
        }

        self.frame += 1;

        Ok(vec![
            //DrawCmd::new(self.tri_verts).indices(self.tri_indices),
            DrawCmd::new(self.line_verts)
                .indices(self.line_indices)
                .shader(self.line_shader)
                .limit(self.line_gb.indices.len() as u32),
        ])
    }

    fn event(&mut self, ctx: &mut Context, platform: &mut Platform, event: Event) -> Result<()> {
        ortho_cam_ctx(ctx, platform);
        idek::close_when_asked(platform, &event);
        Ok(())
    }
}

fn new_grid(rng: &mut Rng, tiles: &[Tile]) -> Array2D<TileSet> {
    let w = 15;
    let mut grid = init_grid(w, w, &tiles);

    for _ in 0..4 {
        let x = rng.gen() as usize % grid.width();
        let y = rng.gen() as usize % grid.height();
        let idx = rng.gen() as usize % tiles.len();

        let tile = &mut grid[(x, y)];
        tile.iter_mut().for_each(|b| *b = false);
        tile[idx] = true;
    }

    grid
}

fn extend_gb(dest: &mut ShapeBuilder, src: &ShapeBuilder, transform: Isometry2<f32>) {
    let base = dest.vertices.len() as u32;
    dest.vertices.extend_from_slice(&src.vertices);
    dest.indices.extend(src.indices.iter().map(|i| i + base));
}

const PATH_WIDTH: f32 = 0.5;
const PATH_HALFW: f32 = PATH_WIDTH / 2.;
const PATH_MIN: f32 = 0.5 - PATH_HALFW;
const PATH_MAX: f32 = 0.5 + PATH_HALFW;

fn cons_shape(f: fn(&mut ShapeBuilder)) -> ShapeBuilder {
    let mut gb = ShapeBuilder::new();
    f(&mut gb);
    gb
}

/**
```txt
+--> +X
|
V
+Y

+------+
| c____e
| | d__f
| | |  |
+-a b--+
<-> PATH_WIDTH
```
*/
fn path_right(gb: &mut ShapeBuilder) {
    let a = gb.push_vertex([PATH_MIN, 1., 0.]);
    let b = gb.push_vertex([PATH_MAX, 1., 0.]);
    let c = gb.push_vertex([PATH_MIN, PATH_MIN, 0.]);
    let d = gb.push_vertex([PATH_MAX, PATH_MAX, 0.]);
    let e = gb.push_vertex([1., PATH_MIN, 0.]);
    let f = gb.push_vertex([1., PATH_MAX, 0.]);

    gb.push_indices(&[a, c, c, e, b, d, d, f])
}

/**
```txt
+--> +X
|
V
+Y

+-c  d-+
| |  | |
| |  | |
| |  | |
+-a  b-+
  <--> PATH_WIDTH
```
*/
fn path_straight(gb: &mut ShapeBuilder) {
    let a = gb.push_vertex([PATH_MIN, 1., 0.]);
    let b = gb.push_vertex([PATH_MAX, 1., 0.]);
    let c = gb.push_vertex([PATH_MIN, 0., 0.]);
    let d = gb.push_vertex([PATH_MAX, 0., 0.]);

    gb.push_indices(&[a, c, b, d]);
}

/**
```txt
+--> +X
|
V
+Y

+------+
| c--d |
| |  | |
| |  | |
+-a  b-+
<-> PATH_WIDTH
```
*/
fn path_cap(gb: &mut ShapeBuilder) {
    let a = gb.push_vertex([PATH_MIN, 1., 0.]);
    let b = gb.push_vertex([PATH_MAX, 1., 0.]);
    let c = gb.push_vertex([PATH_MIN, PATH_MIN, 0.]);
    let d = gb.push_vertex([PATH_MAX, PATH_MIN, 0.]);

    gb.push_indices(&[a, c, c, d, d, b]);
}

/**
```txt
+--> +X
|
V
+Y

+-c  d-+
| |  e g
| |
| |  f h
+-a  b-+
<-> PATH_WIDTH
```
*/
fn path_3way(gb: &mut ShapeBuilder) {
    let a = gb.push_vertex([PATH_MIN, 1., 0.]);
    let b = gb.push_vertex([PATH_MAX, 1., 0.]);
    let c = gb.push_vertex([PATH_MIN, 0., 0.]);
    let d = gb.push_vertex([PATH_MAX, 0., 0.]);

    let e = gb.push_vertex([PATH_MAX, PATH_MIN, 0.]);
    let f = gb.push_vertex([PATH_MAX, PATH_MAX, 0.]);
    let g = gb.push_vertex([1., PATH_MIN, 0.]);
    let h = gb.push_vertex([1., PATH_MAX, 0.]);

    gb.push_indices(&[c, a, d, e, e, g, f, h, f, b]);
}

/**
```txt
+--> +X
|
V
+Y

+-c  d-+
e-i  j-g

f-k  l-h
+-a  b-+
<-> PATH_WIDTH
```
*/
fn path_4way(gb: &mut ShapeBuilder) {
    let a = gb.push_vertex([PATH_MIN, 1., 0.]);
    let b = gb.push_vertex([PATH_MAX, 1., 0.]);
    let c = gb.push_vertex([PATH_MIN, 0., 0.]);
    let d = gb.push_vertex([PATH_MAX, 0., 0.]);

    let e = gb.push_vertex([0., PATH_MIN, 0.]);
    let f = gb.push_vertex([0., PATH_MAX, 0.]);
    let g = gb.push_vertex([1., PATH_MIN, 0.]);
    let h = gb.push_vertex([1., PATH_MAX, 0.]);

    let i = gb.push_vertex([PATH_MIN, PATH_MIN, 0.]);
    let j = gb.push_vertex([PATH_MAX, PATH_MIN, 0.]);
    let k = gb.push_vertex([PATH_MIN, PATH_MAX, 0.]);
    let l = gb.push_vertex([PATH_MAX, PATH_MAX, 0.]);

    gb.push_indices(&[a, k, b, l, g, j, h, l, c, i, d, j, e, i, f, k]);
}

/// Return a camera prefix matrix which keeps (-1, 1) on XY visible and at a 1:1 aspect ratio
pub fn ortho_cam((width, height): (u32, u32)) -> Matrix4<f32> {
    let (width, height) = (width as f32, height as f32);
    let (znear, zfar) = (-1., 1.);
    match width < height {
        true => Matrix4::new_orthographic(0., 1., 0., height / width, znear, zfar),
        false => Matrix4::new_orthographic(0., width / height, 0., 1., znear, zfar),
    }
}

/// Same as `simple_ortho_cam` but using the builtin inputs
pub fn ortho_cam_ctx(ctx: &mut Context, platform: &mut Platform) {
    if !platform.is_vr() {
        ctx.set_camera_prefix(ortho_cam(ctx.screen_size()));
    }
}

pub fn draw_tile_grid(
    gb: &mut ShapeBuilder,
    grid: &Array2D<TileSet>,
    tiles: &[Tile],
    dirty: HashSet<(usize, usize)>,
) {
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            if dirty.contains(&(x, y)) {
                gb.set_color([1., 0., 0.]);
            } else {
                gb.set_color([1.; 3]);
            }

            let set = &grid[(x, y)];

            let maxdim = grid.width().max(grid.height());
            let [x, y] = [x, y].map(|v| v as f32 / maxdim as f32);

            gb.push_tf(Similarity3::from_isometry(
                Isometry3::translation(x, y, 0.),
                1. / maxdim as f32,
            ));

            draw_tile(gb, set, tiles);

            gb.pop_tf();
        }
    }
}

pub fn draw_tile(gb: &mut ShapeBuilder, set: &TileSet, tiles: &[Tile]) {
    let total = set.iter().filter(|p| **p).count();
    let side_len = ceil_pow2(total);

    'outer: for y in 0..side_len {
        for x in 0..side_len {
            let scale = 1. / side_len as f32;

            let n = x + y * side_len;

            let [x, y] = [x, y].map(|v| v as f32 / side_len as f32);

            let idx = set
                .iter()
                .enumerate()
                .filter(|(_, p)| **p)
                .skip(n)
                .find_map(|(i, p)| p.then(|| i));

            let idx = match idx {
                None => break 'outer,
                Some(i) => i,
            };

            gb.push_tf(pos_scale2d(x, y, scale));
            gb.append(&tiles[idx].art);
            gb.pop_tf();
        }
    }
}

fn pos_scale2d(x: f32, y: f32, scale: f32) -> Similarity3<f32> {
    Similarity3::from_isometry(Isometry3::translation(x, y, 0.), scale)
}

pub fn draw_background_grid(gb: &mut ShapeBuilder, width: usize, height: usize) {
    for i in 0..=width {
        let x = i as f32 / width as f32;
        let top = gb.push_vertex([x, 0., 0.]);
        let bottom = gb.push_vertex([x, 1., 0.]);
        gb.push_indices(&[top, bottom]);
    }

    for i in 0..=height {
        let y = i as f32 / height as f32;
        let left = gb.push_vertex([0., y, 0.]);
        let right = gb.push_vertex([1., y, 0.]);
        gb.push_indices(&[left, right]);
    }
}

pub fn ceil_pow2(v: usize) -> usize {
    (v as f32).sqrt().ceil() as usize
}

pub fn draw_solver(gb: &mut ShapeBuilder, solver: &Solver) {
    gb.set_color([1., 0.2, 0.2]);
    draw_background_grid(gb, solver.grid().width(), solver.grid().height());

    gb.set_color([1.; 3]);
    let dirty: HashSet<(usize, usize)> = solver.dirty().iter().copied().collect();
    draw_tile_grid(gb, solver.grid(), solver.tiles(), dirty);
}

pub fn grid_entropy(grid: &Grid) -> usize {
    grid.data().iter().map(count_tileset).sum()
}
