from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 anchor, found {count}")
    return text.replace(old, new, 1)


p = Path("src/generation_store.rs")
s = p.read_text()

s = once(
    s,
    "use std::fs::{self, File};\nuse std::io::{self, Read, Write};\nuse std::path::Path;\nuse std::time::{SystemTime, UNIX_EPOCH};\n",
    "use std::fs::{self, File, OpenOptions};\nuse std::io::{self, Read, Write};\nuse std::path::{Path, PathBuf};\nuse std::sync::atomic::{AtomicU64, Ordering};\n",
    "imports",
)

s = once(
    s,
    'const CURRENT_FILE: &str = "CURRENT";\n',
    'const CURRENT_FILE: &str = "CURRENT";\nconst PREVIOUS_CURRENT_FILE: &str = ".CURRENT.previous";\nconst MAX_TEMP_ATTEMPTS: usize = 1024;\nstatic NEXT_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);\n',
    "publication constants",
)

old_staging = '''    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("system clock is before UNIX_EPOCH"))?
        .as_nanos();
    let staging = root.join(format!(
        ".{generation_name}-{}-{nonce}.tmp",
        std::process::id()
    ));
    fs::create_dir(&staging)?;
'''
new_staging = '''    let staging = create_unique_staging_dir(root, &generation_name)?;
'''
s = once(s, old_staging, new_staging, "staging nonce")

old_open = '''    let current = root.join(CURRENT_FILE);
    if fs::symlink_metadata(&current).is_ok() {
        crate::fileguard::guard_not_symlink("hybrid CURRENT", &current)?;
        let raw = read_small_text(&current, MAX_CURRENT_BYTES, "hybrid CURRENT")?;
        let (name, expected_manifest_hash) = parse_current(&raw)?;
        return open_generation(
            root,
            &name,
            dim,
            Some(&expected_manifest_hash),
            expected_fingerprint,
        );
    }

    // Crash recovery: if the pointer has not yet been published (or a platform
    // had to remove it before replacement), the highest immutable generation is
    // still self-contained and can be validated. Hidden staging directories are
    // deliberately ignored by `highest_generation`.
    if let Some(generation) = highest_generation(root)? {
        return open_generation(
            root,
            &generation_name(generation),
            dim,
            None,
            expected_fingerprint,
        );
    }
'''
new_open = '''    let current = root.join(CURRENT_FILE);
    if fs::symlink_metadata(&current).is_ok() {
        return open_pointer(
            root,
            &current,
            "hybrid CURRENT",
            dim,
            expected_fingerprint,
        );
    }

    #[cfg(not(unix))]
    {
        // Windows publication temporarily moves the previously published pointer
        // aside because std::fs::rename cannot portably replace an existing file.
        // If a crash happens in that narrow window, only that *previous pointer*
        // is authoritative. An immutable generation without a pointer is never
        // promoted merely because it has the largest number.
        let previous = root.join(PREVIOUS_CURRENT_FILE);
        if fs::symlink_metadata(&previous).is_ok() {
            return open_pointer(
                root,
                &previous,
                "hybrid previous CURRENT",
                dim,
                expected_fingerprint,
            );
        }
    }

    if highest_generation(root)?.is_some() {
        return Err(invalid(
            "hybrid generation exists but no published CURRENT pointer is present",
        ));
    }
'''
s = once(s, old_open, new_open, "published-only open")

marker = '''fn open_generation(
    root: &Path,
'''
open_pointer = '''fn open_pointer(
    root: &Path,
    pointer: &Path,
    what: &str,
    dim: usize,
    expected_fingerprint: Option<&GenerationFingerprint>,
) -> io::Result<HybridMemory> {
    crate::fileguard::guard_not_symlink(what, pointer)?;
    let raw = read_small_text(pointer, MAX_CURRENT_BYTES, what)?;
    let (name, expected_manifest_hash) = parse_current(&raw)?;
    open_generation(
        root,
        &name,
        dim,
        Some(&expected_manifest_hash),
        expected_fingerprint,
    )
}

'''
s = once(s, marker, open_pointer + marker, "open pointer helper")

old_publish_head = '''fn publish_current(root: &Path, generation: &str, manifest_sha256: &str) -> io::Result<()> {
    let current = root.join(CURRENT_FILE);
    reject_symlink_if_present("hybrid CURRENT", &current)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("system clock is before UNIX_EPOCH"))?
        .as_nanos();
    let tmp = root.join(format!(".CURRENT-{}-{nonce}.tmp", std::process::id()));
    let body =
        format!("{CURRENT_MAGIC}\\ngeneration={generation}\\nmanifest_sha256={manifest_sha256}\\n");
    write_synced(&tmp, body.as_bytes())?;
'''
new_publish_head = '''fn publish_current(root: &Path, generation: &str, manifest_sha256: &str) -> io::Result<()> {
    let current = root.join(CURRENT_FILE);
    reject_symlink_if_present("hybrid CURRENT", &current)?;
    let body =
        format!("{CURRENT_MAGIC}\\ngeneration={generation}\\nmanifest_sha256={manifest_sha256}\\n");
    let tmp = write_unique_temp_file(root, "CURRENT", body.as_bytes())?;
'''
s = once(s, old_publish_head, new_publish_head, "current temp nonce")

old_windows = '''        // Windows does not guarantee replacement of an existing destination.
        // The generation directory is already durable, so a crash in this tiny
        // pointer window is recovered by scanning immutable generations on open.
        let backup = root.join(".CURRENT.previous");
        if current.exists() {
            let _ = fs::remove_file(&backup);
            fs::rename(&current, &backup)?;
        }
        if let Err(err) = fs::rename(&tmp, &current) {
            if backup.exists() {
                let _ = fs::rename(&backup, &current);
            }
            return Err(err);
        }
        let _ = fs::remove_file(&backup);
'''
new_windows = '''        // Windows does not guarantee replacement of an existing destination.
        // Preserve the *previously published pointer* during the replacement
        // window. Recovery may use this pointer, never an unreferenced generation.
        let backup = root.join(PREVIOUS_CURRENT_FILE);
        reject_symlink_if_present("hybrid previous CURRENT", &backup)?;
        if current.exists() {
            if backup.exists() {
                fs::remove_file(&backup)?;
            }
            fs::rename(&current, &backup)?;
        }
        if let Err(err) = fs::rename(&tmp, &current) {
            if backup.exists() {
                let _ = fs::rename(&backup, &current);
            }
            let _ = fs::remove_file(&tmp);
            return Err(err);
        }
        if backup.exists() {
            fs::remove_file(&backup)?;
        }
'''
s = once(s, old_windows, new_windows, "windows publication recovery")

# Clean up a temporary pointer if POSIX publication fails.
old_unix = '''    #[cfg(unix)]
    {
        // POSIX rename replaces the old file atomically.
        fs::rename(&tmp, &current)?;
    }
'''
new_unix = '''    #[cfg(unix)]
    {
        // POSIX rename replaces the old file atomically. The generation becomes
        // authoritative only at this pointer publication step.
        if let Err(err) = fs::rename(&tmp, &current) {
            let _ = fs::remove_file(&tmp);
            return Err(err);
        }
    }
'''
s = once(s, old_unix, new_unix, "unix publication cleanup")

# Add clock-independent, collision-safe temporary constructors.
marker = '''fn highest_generation(root: &Path) -> io::Result<Option<u64>> {
'''
helpers = '''fn create_unique_staging_dir(root: &Path, generation_name: &str) -> io::Result<PathBuf> {
    for _ in 0..MAX_TEMP_ATTEMPTS {
        let nonce = NEXT_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            ".{generation_name}-{}-{nonce}.tmp",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique hybrid staging directory",
    ))
}

fn write_unique_temp_file(root: &Path, stem: &str, bytes: &[u8]) -> io::Result<PathBuf> {
    for _ in 0..MAX_TEMP_ATTEMPTS {
        let nonce = NEXT_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            ".{stem}-{}-{nonce}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                if let Err(err) = file.write_all(bytes).and_then(|_| file.sync_all()) {
                    let _ = fs::remove_file(&path);
                    return Err(err);
                }
                return Ok(path);
            }
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique hybrid pointer temporary file",
    ))
}

'''
s = once(s, marker, helpers + marker, "temp helper insertion")

# Replace the old recovery test with fail-closed publication semantics.
old_test = '''    #[test]
    fn missing_current_recovers_highest_immutable_generation_and_ignores_staging() {
        let root = temp_store("recovery");
        let memory = populated(23);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();
        fs::remove_file(root.join(CURRENT_FILE)).unwrap();
        fs::create_dir(root.join(".generation-99999999999999999999-dead.tmp")).unwrap();

        let reopened = open(root.to_string_lossy().as_ref(), 4).unwrap();
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.default_shortlist, 23);
        let _ = fs::remove_dir_all(root);
    }
'''
new_test = '''    #[test]
    fn missing_current_never_promotes_an_unpublished_generation() {
        let root = temp_store("unpublished");
        let memory = populated(23);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();
        fs::remove_file(root.join(CURRENT_FILE)).unwrap();
        fs::create_dir(root.join(".generation-99999999999999999999-dead.tmp")).unwrap();

        let err = open(root.to_string_lossy().as_ref(), 4)
            .err()
            .expect("generation without CURRENT was silently promoted");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("no published CURRENT"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn orphan_generation_does_not_override_the_published_pointer() {
        let root = temp_store("orphan");
        let memory = populated(29);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();

        let orphan = root.join(generation_name(2));
        fs::create_dir(&orphan).unwrap();
        for name in [TREE_FILE, SKETCH_FILE, MANIFEST_FILE] {
            fs::write(orphan.join(name), b"not published").unwrap();
        }

        let reopened = open(root.to_string_lossy().as_ref(), 4).unwrap();
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.default_shortlist, 29);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn temp_allocation_does_not_depend_on_wall_clock() {
        let root = temp_store("temp-nonce");
        fs::create_dir_all(&root).unwrap();
        let first = create_unique_staging_dir(&root, "generation-00000000000000000001").unwrap();
        let second = create_unique_staging_dir(&root, "generation-00000000000000000001").unwrap();
        assert_ne!(first, second);
        assert!(first.is_dir());
        assert!(second.is_dir());
        let _ = fs::remove_dir_all(root);
    }
'''
s = once(s, old_test, new_test, "recovery regression")

p.write_text(s)
