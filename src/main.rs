use idek::{prelude::*, IndexBuffer};
use idek_basics::{idek::{self, nalgebra::Isometry2}, GraphicsBuilder};
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

    camera: MultiPlatformCamera,
}

impl App for CubeDemo {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let line_gb = rainbow_cube();
        let tri_gb = rainbow_cube();

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
            DrawCmd::new(self.tri_verts).indices(self.tri_indices),
            DrawCmd::new(self.line_verts).indices(self.line_indices).shader(self.line_shader),
        ])
    }

    fn event(&mut self, ctx: &mut Context, platform: &mut Platform, event: Event) -> Result<()> {
        self.camera.handle_event(&event);
        idek::simple_ortho_cam_ctx(ctx, platform);
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

fn left_turn(gb: &mut GraphicsBuilder) {
    todo!()
}
