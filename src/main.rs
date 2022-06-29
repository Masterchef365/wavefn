use idek::{prelude::*, IndexBuffer};
use idek_basics::{
    idek::{self, nalgebra::{Isometry2, Matrix4}},
    GraphicsBuilder,
};
use wavefn::{Solver, Tile};

fn main() -> Result<()> {
    launch::<_, CubeDemo>(Settings::default().vr_if_any_args())
}

struct CubeDemo {
    line_verts: VertexBuffer,
    line_indices: IndexBuffer,
    line_gb: GraphicsBuilder,
    line_shader: Shader,

    tri_verts: VertexBuffer,
    tri_indices: IndexBuffer,
    tri_gb: GraphicsBuilder,
}

impl App for CubeDemo {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let mut line_gb = GraphicsBuilder::new();
        let mut tri_gb = GraphicsBuilder::new();

        //path_right(&mut line_gb, [1.; 3]);
        //path_straight(&mut line_gb, [1.; 3]);
        path_4way(&mut line_gb, [1.; 3]);
        path_right(&mut tri_gb, [1.; 3]);

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

fn draw_solver(solver: &Solver, tiles: &[Tile]) -> GraphicsBuilder {
    let mut gb = GraphicsBuilder::new();
    let grid = solver.grid();
    todo!()
}

fn extend_gb(dest: &mut GraphicsBuilder, src: &GraphicsBuilder, transform: Isometry2<f32>) {
    let base = dest.vertices.len() as u32;
    dest.vertices.extend_from_slice(&src.vertices);
    dest.indices.extend(src.indices.iter().map(|i| i + base));
}


const PATH_WIDTH: f32 = 0.5;
const PATH_HALFW: f32 = PATH_WIDTH / 2.;
const PATH_MIN: f32 = 0.5 - PATH_HALFW;
const PATH_MAX: f32 = 0.5 + PATH_HALFW;

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
fn path_right(gb: &mut GraphicsBuilder, color: [f32; 3]) {
    let mut addv = |x, y| gb.push_vertex(Vertex::new([x, y, 0.], color));

    let a = addv(PATH_MIN, 1.);
    let b = addv(PATH_MAX, 1.);
    let c = addv(PATH_MIN, PATH_MIN);
    let d = addv(PATH_MAX, PATH_MAX);
    let e = addv(1., PATH_MIN);
    let f = addv(1., PATH_MAX);

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
fn path_straight(gb: &mut GraphicsBuilder, color: [f32; 3]) {
    let mut addv = |x, y| gb.push_vertex(Vertex::new([x, y, 0.], color));

    let a = addv(PATH_MIN, 1.);
    let b = addv(PATH_MAX, 1.);
    let c = addv(PATH_MIN, 0.);
    let d = addv(PATH_MAX, 0.);

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
fn path_4way(gb: &mut GraphicsBuilder, color: [f32; 3]) {
    let mut addv = |x, y| gb.push_vertex(Vertex::new([x, y, 0.], color));
    let a = addv(PATH_MIN, 1.);
    let b = addv(PATH_MAX, 1.);
    let c = addv(PATH_MIN, 0.);
    let d = addv(PATH_MAX, 0.);

    let e = addv(0., PATH_MIN);
    let f = addv(0., PATH_MAX);
    let g = addv(1., PATH_MIN);
    let h = addv(1., PATH_MAX);

    let i = addv(PATH_MIN, PATH_MIN);
    let j = addv(PATH_MAX, PATH_MIN);
    let k = addv(PATH_MIN, PATH_MAX);
    let l = addv(PATH_MAX, PATH_MAX);

    gb.push_indices(&[
        a, k,
        b, l,
        g, j,
        h, l,
        c, i,
        d, j,
        e, i,
        f, k
    ]);
}

/// Return a camera prefix matrix which keeps (-1, 1) on XY visible and at a 1:1 aspect ratio
pub fn ortho_cam((width, height): (u32, u32)) -> Matrix4<f32> {
    let (width, height) = (width as f32, height as f32);
    let (znear, zfar) = (-1., 1.);
    match width < height {
        true => Matrix4::new_orthographic(0., 1., 0., height/width, znear, zfar),
        false => Matrix4::new_orthographic(0., width/height, 0., 1., znear, zfar),
    }
}

/// Same as `simple_ortho_cam` but using the builtin inputs
pub fn ortho_cam_ctx(ctx: &mut Context, platform: &mut Platform) {
    if !platform.is_vr() {
        ctx.set_camera_prefix(ortho_cam(ctx.screen_size()));
    }
}
