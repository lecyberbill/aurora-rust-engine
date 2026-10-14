p = "D:/Projet/UI/Cargo.toml"
with open(p, "r", encoding="utf-8") as f:
    text = f.read()

new_text = text.replace('version = "0.1.0"', 'version = "0.1.1"')

with open(p, "w", encoding="utf-8") as f:
    f.write(new_text)

print("Successfully bumped grio to version 0.1.1!")
