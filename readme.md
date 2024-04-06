
---



## Overview
This Flask application utilizes machine learning models from the Hugging Face `transformers` library to generate embeddings for input texts and find the most similar text in a predefined knowledge base. It leverages the power of pre-trained models for understanding and comparing text data, making it useful for applications requiring text similarity, question answering, or recommendation systems.

## Features
- **Generate Embeddings**: Converts input texts into dense vector representations using a pre-trained model.
- **Text Similarity**: Finds the most similar text in the knowledge base to the input query.
- **REST API**: Offers a simple RESTful API endpoint for processing text similarity queries.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://your-repository-url.git
    cd your-project-directory
    ```

2. **Setup Environment**

    Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    Install the required packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**

    Start the Flask application locally:

    ```bash
    python main.py
    ```

    Or use the provided `start.sh` script:

    ```bash
    ./start.sh
    ```

## Usage

To test the application, use the `/chat` endpoint to send a POST request with a JSON body containing the text for which you want to find a similar match.

### Testing with Postman

1. **Method**: POST
2. **URL**: `http://localhost:5000/chat`
3. **Headers**:
    - Content-Type: application/json
4. **Body**:
    ```json
    {
        "text": ["Your text here"]
    }
    ```

Replace `"Your text here"` with the actual text you want to find similarities for.

### Response

The response will be a JSON object containing the most similar text from the knowledge base:

```json
{
    "response": ["Similar text response"]
}
```

## Docker Deployment

Refer to the provided `Dockerfile` for containerizing and deploying the application using Docker.

## Note

Ensure you have Docker installed and configured if you plan to deploy the application using a Docker container. For deploying on cloud platforms like AWS EC2, additional setup for security and environment variables might be required.

---
