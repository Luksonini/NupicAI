fn main() {
    println!("cargo:rerun-if-env-changed=NUPICAI_SERVER_URL");
    tauri_build::build()
}
