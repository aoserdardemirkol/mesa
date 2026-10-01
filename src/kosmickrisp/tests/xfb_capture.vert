#version 450

layout(xfb_buffer = 0, xfb_stride = 16) out;
layout(location = 0, xfb_offset = 0) out vec4 captured;

void main()
{
   uint id = uint(gl_VertexIndex);
   captured = vec4(float(id * 4 + 1), float(id * 4 + 2),
                   float(id * 4 + 3), float(id * 4 + 4));
   gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
