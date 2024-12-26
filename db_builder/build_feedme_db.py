import json
import itertools
import numpy as np
from transformers import CLIPTokenizer
from hailo_sdk_client import ClientRunner, InferenceContext


class FeedMeDB:
    def __init__(self, har_path, config_file_path, tokenizer_path):
        print("Initializing FeedMeDB...")
        runner = ClientRunner(har=har_path)
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
            self.emulation_model = runner.get_keras_model(ctx)

        config = np.load(config_file_path) 
        self.text_projection = config["projection_layer"]
        self.text_embeddings = config["text_embeddings"]
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        print("FeedMeDB initialized and emulation model is ready in SDK_QUANTIZED context.")
    
    def text_encoder_preprocessing(self, prompt):
        return self.tokenizer(
            prompt,
            padding="max_length",   # Pad to the model's maximum sequence length
            truncation=True,        # Truncate if the text exceeds the maximum length
            max_length=77,          # CLIP typically uses 77 tokens for text
            return_tensors="tf"     # Return PyTorch tensors (use "tf" for TensorFlow))
        )

    def text_encoding_postprocessing(self, logits, length):
        result = logits @ self.text_projection.T
        result = result[:length]
        result = result/ np.linalg.norm(result, axis=-1, keepdims=True)
        return np.mean(result, axis=-2)

    def get_embeds(self, text):
        preproc = self.text_encoder_preprocessing(text)
        tokens = np.array(preproc['input_ids'])
        length = np.sum(preproc['attention_mask'] > 0) - 1
        x = np.expand_dims(self.text_embeddings[tokens], axis=0)
        outputs = self.emulation_model(x)
        embeds = self.text_encoding_postprocessing(outputs, length)
        return embeds[0, 0]

    def build_encoding_db(self, json_file_path, output_db_path):
        feedme_db = {}
        menu_data = json.load(open(json_file_path))
        for menu_item in menu_data['menu']:
            # get the encoding for the menu item
            major_key = menu_item['type']
            feedme_db[major_key] = {}
            feedme_db[major_key]['encoding'] = self.get_embeds(major_key)
            feedme_db[major_key]['sub_items'] = {}
            
            for sub_item in menu_item['items']:
                # get the encoding for the sub-item
                minor_key = f"{sub_item['name']}: {sub_item['description']}"
                feedme_db[major_key]['sub_items'][minor_key] = {}
                feedme_db[major_key]['sub_items'][minor_key]['encoding'] = self.get_embeds(minor_key)
        
                # build list of all combinations of changes possible for the sub-item
                combinations = []
                optional_changes = sub_item['optional_changes']
                for i in range(1, len(optional_changes) + 1):
                    combinations.extend(itertools.combinations(optional_changes, i))
                combinations_keys = [', '.join(list(comb)) for comb in combinations]
                combinations_keys = [f'{minor_key}, {x}' for x in combinations_keys]
                combinations_encodings = {x: self.get_embeds(x) for x in combinations_keys}
                feedme_db[major_key]['sub_items'][minor_key]['possible_changes'] = combinations_encodings
                print(f"Built encodings for {major_key}:{minor_key} and its possible changes.")

        # save encodings for items and sub-items in the menu data object
        with open(output_db_path, 'wb') as f:
            np.savez(f, **feedme_db)

    
if __name__ == "__main__":
    har_path = 'compile/clip_text_encoder_vit_large.har'
    config_file_path = '/data/data/adk_models_files/models_files/clip/vit_large/coco_10xtd/2024-08-25/openai-clip-vit-large-patch14.npz'
    tokenizer_path = "openai/clip-vit-large-patch14"
    feedme_db = FeedMeDB(har_path, config_file_path, tokenizer_path)
    feedme_db.build_encoding_db("burger_menu.json", "feedme_db_burgers.npz")
    print("Done building FeedMeDB.")
