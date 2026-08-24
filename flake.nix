{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      commonPackages = with pkgs; [
        uv
        clang # Required by Inductor.
      ];
      commonEnv = {
        UV_PYTHON = "3.12";
      };
      mkComfyShell =
        {
          packages ? [ ],
          env ? { },
          shellHook ? "",
          torchBackend,
        }:
        let
          venv = ".venv-${torchBackend}";
        in
        pkgs.mkShell {
          packages = commonPackages ++ packages;
          env = commonEnv // env // { UV_PROJECT_ENVIRONMENT = venv; };
          shellHook = ''
            export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torch/inductor"
            uv venv --allow-existing "$UV_PROJECT_ENVIRONMENT"
            ${shellHook}
            uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" torch torchvision torchaudio --torch-backend ${torchBackend}
            uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -r comfyui/requirements.txt -r comfyui/manager_requirements.txt
            source "$UV_PROJECT_ENVIRONMENT/bin/activate"
          '';
        };
      cuda = mkComfyShell {
        torchBackend = "cu130";
      };
      rocm = mkComfyShell {
        torchBackend = "rocm7.2";
      };
      xpu =
        with pkgs;
        mkComfyShell {
          packages = [
            level-zero
            openssl
          ];
          env = {
            OCL_ICD_VENDORS = "${intel-compute-runtime}/etc/OpenCL/vendors/intel-neo.icd";
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              level-zero
              intel-compute-runtime
              intel-compute-runtime.drivers
              intel-graphics-compiler
            ];
            LD_PRELOAD = "${ocl-icd}/lib/libOpenCL.so.1";
          };
          shellHook = ''
            ln -sf ${intel-compute-runtime}/bin/ocloc-* "$UV_PROJECT_ENVIRONMENT/bin/ocloc"
          '';
          torchBackend = "xpu";
        };
      cpu = mkComfyShell {
        torchBackend = "cpu";
      };
    in
    {
      devShells.${system} = {
        default = cpu;
        inherit
          cuda
          rocm
          xpu
          cpu
          ;
      };
    };
}
