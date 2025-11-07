{
    description = "Dev shell for helix-vs-gemini benchmark";

    inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

    outputs = { self, nixpkgs }:
      let
        system = "x86_64-linux";
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.${system}.default = pkgs.mkShell {
          packages = with pkgs; [
            python313             # same Python you’re using
            uv
            stdenv.cc.cc.lib      # provides libstdc++.so.6
            pkg-config
            openssl
          ];

          shellHook = ''
            export UV_SYSTEM_PYTHON=1
            echo "Dev shell ready (python, uv, libstdc++)."
          '';
        };
      };
  }
