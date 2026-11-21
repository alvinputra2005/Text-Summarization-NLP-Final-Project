from src.summarization import INDONESIAN, generate_mbart, load_mbart, model_reference


def main() -> None:
    reference = model_reference()
    print(f"Memuat checkpoint: {reference}", flush=True)
    tokenizer, model = load_mbart(reference)
    revision = getattr(model.config, "_commit_hash", None)
    if revision:
        print(f"Revisi checkpoint: {revision}")
    language_id = tokenizer.lang_code_to_id[INDONESIAN]
    text = "Gunakan dropna untuk menghapus baris dengan harga kosong pada data penjualan."
    inputs = tokenizer(text, return_tensors="pt")
    if tokenizer.src_lang != INDONESIAN or tokenizer.tgt_lang != INDONESIAN:
        raise RuntimeError("Bahasa sumber dan target harus id_ID.")
    if inputs["input_ids"][0, 0].item() != language_id:
        raise RuntimeError("Input tidak diawali token bahasa id_ID.")
    print(f"Tokenizer OK: {inputs['input_ids'].shape[-1]} token input ({INDONESIAN})", flush=True)

    output = generate_mbart(inputs, tokenizer, model, max_new_tokens=40)
    if output[0, 1].item() != language_id:
        raise RuntimeError("Keluaran tidak diawali token bahasa id_ID setelah token pembuka dekoder.")
    print(f"Model OK: {output.shape[-1]} token output pada {model.device}")
    print("Contoh keluaran:", tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
