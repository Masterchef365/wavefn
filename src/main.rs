use idek::{prelude::*, IndexBuffer};
use idek_basics::{
    idek::{
        self,
        nalgebra::{Isometry2, Isometry3, Matrix4, Similarity3},
    },
    Array2D, ShapeBuilder,
};
use wavefn::{apply_symmetry, compile_tiles, Shape, Solver, Symmetry, Tile, TileSet};

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
}

const CONN_WALL: u32 = 0;
const CONN_PATH: u32 = 1;

impl App for CubeDemo {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let mut line_gb = ShapeBuilder::new();
        let mut tri_gb = ShapeBuilder::new();

        let mut shapes = vec![];

        /*
        shapes.push(Shape {
            art: cons_shape(path_right),
            conn: [CONN_PATH, CONN_PATH, CONN_WALL, CONN_WALL],
            //weight: 1.,
        });
        shapes.push(Shape {
            art: cons_shape(path_straight),
            conn: [CONN_WALL, CONN_PATH, CONN_WALL, CONN_PATH],
            //weight: 1.,
        });
        shapes.push(Shape {
            art: cons_shape(path_4way),
            conn: [CONN_PATH; 4],
            //weight: 1.,
        });
        */

        // Right turn
        shapes.extend(apply_symmetry(
            &Shape {
                art: cons_shape(path_right),
                conn: [CONN_PATH, CONN_PATH, CONN_WALL, CONN_WALL],
                //weight: 1.,
            },
            Symmetry::Rot4,
        ));

        // Straight path
        shapes.extend(apply_symmetry(
            &Shape {
                art: cons_shape(path_straight),
                conn: [CONN_WALL, CONN_PATH, CONN_WALL, CONN_PATH],
                //weight: 1.,
            },
            Symmetry::Rot2,
        ));

        // 4-way path
        shapes.push(Shape {
            art: cons_shape(path_4way),
            conn: [CONN_PATH; 4],
            //weight: 1.,
        });

        let tiles = compile_tiles(&shapes);
        let solver = Solver::new(tiles, 10, 10);

        line_gb.set_color([1., 0.2, 0.2]);
        draw_background_grid(&mut line_gb, solver.grid().width(), solver.grid().height());

        line_gb.set_color([1.; 3]);
        draw_tile_grid(&mut line_gb, solver.grid(), solver.tiles());

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
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        ctx.update_vertices(self.tri_verts, &self.tri_gb.vertices)?;
        ctx.update_vertices(self.line_verts, &self.line_gb.vertices)?;

        Ok(vec![
            //DrawCmd::new(self.tri_verts).indices(self.tri_indices),
            DrawCmd::new(self.line_verts)
                .indices(self.line_indices)
                .shader(self.line_shader),
        ])
    }

    fn event(&mut self, ctx: &mut Context, platform: &mut Platform, event: Event) -> Result<()> {
        ortho_cam_ctx(ctx, platform);
        idek::close_when_asked(platform, &event);
        Ok(())
    }
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
  <-> PATH_WIDTH
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

pub fn draw_tile_grid(gb: &mut ShapeBuilder, grid: &Array2D<TileSet>, tiles: &[Tile]) {
    for y in 0..grid.height() {
        for x in 0..grid.width() {
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

    let scale = 1. / side_len as f32;

    'outer: for y in 0..side_len {
        for x in 0..side_len {
            let idx = x + y * side_len;
            let [x, y] = [x, y].map(|v| v as f32 / side_len as f32);

            if idx >= set.len() {
                break 'outer;
            }

            if set[idx] {
                gb.push_tf(pos_scale2d(x, y, scale));
                gb.append(&tiles[idx].art);
                gb.pop_tf();
            }
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
