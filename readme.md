# FeeMe - Free Text Menu Ordering

Welcome to the **FeeMe** repository! This open-source project enables users to select items from a restaurant menu by writing a free-text prompt instead of manually choosing each item. FeeMe intelligently matches the user's description to the closest items available on the menu.

### Example Use Case
Instead of selecting individual menu items, users can simply type:

```
I would like 2 hamburgers without ketchup, 1 veggie burger, 2 fries, 2 Coca-Colas, and 1 chocolate ice cream.
```

The system will parse this prompt, understand the order, and create an itemized list based on the restaurant's menu. FeeMe also works to match the user's description to the closest available items, ensuring their wishes are met even if exact matches aren't available.

---

## Features
- **Natural Language Understanding:** Parses free-text prompts to extract menu items, quantities, and customizations.
- **Customizable Menus:** Easily adapt the system to any restaurant's menu.
- **Dynamic Handling of Customizations:** Supports common modifications like "without ketchup" or "extra cheese."
- **Smart Matching:** Matches user descriptions to the closest available items on the menu.
- **AI-Powered Matching:** Utilizes **ViT-CLIP**, a state-of-the-art generative AI model, to find and match the most relevant items from the menu.
- **Extensible Framework:** Add new features like drink sizes, combo deals, or dietary restrictions with ease.
- **Example Applications:** Can be integrated into drive-through machines, automatic GUI-based ordering kiosks, or mobile apps for faster food ordering.
- **Future Enhancements:**
  - **Speech-to-Text:** In the future, when Hailo chips support it, we plan to integrate Whisper for speech-to-text functionality. This will allow users to describe their orders verbally, enhancing accessibility and convenience.
  - **Image-Based Search:** We plan to add functionality to search menu items based on food images, providing another intuitive way for users to place orders.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/free-text-menu-ordering.git
   ```
2. Navigate to the project directory:
   ```bash
   cd free-text-menu-ordering
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Before running FeeMe, the restaurant needs to index its menu by providing an input JSON file. A script is provided to assist with this step. Ensure the menu JSON is prepared and indexed before proceeding.
   Json example:
   burger_menu.json
2. Change the JSON menu file in line 82.
3.download the hefs from: <>
4.Start the application:
   ```bash
   python build_feedme_db.py 
   ```
5. Enter your prompt in the text input field or via the command line.
6. Review the parsed order and confirm.

---

## Examples Included
The repository comes with example menus and a GUI to help you get started quickly. Use these examples to see how FeeMe works and adapt them to your own restaurant menu.

---

## Example Prompt to the GUI 
Luanch the GUI : python menu_example.py
```
I would like 1 pizza with extra cheese, 2 garlic breads, and 3 orange juices.
```
**Output:**
```
Order Summary:
- 1 Pizza (extra cheese)
- 2 Garlic Breads
- 3 Orange Juices
```


----

## Video Example
https://www.youtube.com/watch?v=ltCH55-BGdA

---



---

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request with a clear description of your changes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For questions or feedback, feel free to open an issue or contact us at [your-email@example.com].

