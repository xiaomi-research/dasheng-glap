{
  description = "GLAP (Generalized Language Audio Pretraining) development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "glap-dev";

          buildInputs = with pkgs; [
            python3
            uv
            git
            libsndfile
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
            stdenv.cc.cc.lib
            zlib
          ];

          env = {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.libsndfile
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ];
          };

          shellHook = ''
            if [ ! -d .venv ]; then
              uv venv --python python3
            fi
            source .venv/bin/activate
            uv sync
          '';
        };
      }
    );
}
