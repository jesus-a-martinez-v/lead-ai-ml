Here are the instructions to set up and run the project:

1. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows use `env\Scripts\activate`
   ```

2. **Install the dependencies:** 
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up the environment variable:**
   - Create a `.env` file in the project root directory.
   - Add the following line to the `.env` file:
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     ```

4. **Run the eval script:**
   ```sh
   python eval.py
   ```

Make sure to replace `your_openai_api_key_here` with your actual OpenAI API key.

If you want to interact with the model in the UI, run: 
```sh
python main.py
```