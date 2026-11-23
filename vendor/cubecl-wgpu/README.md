# CubeCL WGPU Runtime

[CubeCL](https://github.com/tracel-ai/cubecl) WGPU runtime.

This copy comes from `cubecl-wgpu` 0.12.0-pre.2. It requests 64-bit SPIR-V
address calculations when the native Vulkan device supports storage buffer
bindings larger than 4 GiB. The request is applied when creating the shader
module, including when loading a cached kernel.
[Vulkan extension specification](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_shader_64bit_indexing.html)

## Configuration

You can set `CUBECL_WGPU_MAX_TASKS` to a positive integer that determines how many computing tasks are submitted in batches to the graphics API.

## Platform Support

| Option    | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :-------- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| Metal     | No  | Yes |  No   |  Yes  |   No    |   No    | Yes |  No  |
| Vulkan    | Yes | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| OpenGL    | No  | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| WebGpu    | No  | Yes |  No   |  No   |   No    |   No    | No  | Yes  |
| Dx11/Dx12 | No  | Yes |  No   |  No   |   Yes   |   No    | No  |  No  |
