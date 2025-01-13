# Psychotherapy RAG and Web Agent

Psy AI is an agent that leverages local Ollama Llama 3.1 8b model together with RAG and web capabilites to provide insights based on user input. Its prompting is optimized to psychology, psychotherapy, and psychiatry to help user in most professional yet caring manner.

## How It Works

### Local Ollama Llama 3.1 8b Agent (agent.py)
The Psy AI assistant is built using the locally hosted Ollama Llama 3.1 8b model, which serves as the foundational language model for the system. Tests were conducted using Nvidia Geforce RTX 4090, which provided low latency during model interactions. If you wish, you can substitute the default model with any desired one, whether based on API or local.

### LangGraph Architecture

The internal logic of the agent is based on LangGraph framework. This graph controls the flow of actions, retrievals, and state management. The agent dynamically updates its knowledge and state as it interacts with the user.

![LangGraph Structure](images/graph.png)

### LlamaIndex Retrieval-Augmented Generation
The agent uses LlamaIndex, which powers the RAG process. The assistant utilizes RAG to intelligently retrieve relevant information from local knowledge base documents, ensuring that responses are grounded in accurate, contextually relevant data.

The assistant interacts with a vector database, such as ChromaDB, to access indexed knowledge. This enables the system to answer complex queries by fetching related documents and using them in conjunction with the language model to generate responses. This approach enhances the assistant’s ability to provide precise and well-informed answers based on external knowledge beyond its pre-existing training.

### Tavily
In addition to the RAG system, the assistant is equipped with Tavily Web, allowing it to fetch real-time data from the web when necessary. This integration ensures that the assistant can go beyond its internal knowledge base and offer up-to-date, web-sourced information when answering user questions that require more current or specific data.

### Streamlit Interface

The Streamlit app provides a conversational interface. Users input their queries, and the agent processes and responds accordingly. The app maintains session state, keeping track of the conversation history.

![LangGraph Structure](images/UI.png)

## App Launch

## Prerequisites
- Python 3.12 or higher
- UV installed
- Docker
- Llama model being served
- Embedding model being served
- Generate [Tavily API key](https://app.tavily.com/)

## Local Model Serving

### Build Docker Image
```bash
# Build the Docker image
docker build -t psy-model .

# Change the Docker tag (if needed)
docker tag psy-model:latest psyserviceregistry.azurecr.io/psy-model
```

### Launch Docker Container
```bash
# Run the Docker container with GPU support and exposed ports
docker run --runtime=nvidia --gpus=all -p 8000:8000 -p 11434:11434 psy-model
```

## Azure Cloud Deployment

### Docker Push
```bash
# Login to Azure Container Registry
az acr login --name <psyserviceregistry>

# Push the Docker image to Azure
docker push psyserviceregistry.azurecr.io/psy-model
```

### Azure AKS Deployment
1. Create a Kubernetes cluster with a GPU pool node.
2. Install the NVIDIA device plugin on the GPU node:
   ```bash
   az aks command invoke \
       --resource-group myResourceGroup \
       --name myAKSCluster \
       --command "kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml"
   ```
3. Restart the node.
4. Label the GPU pool node as `pool=gpunode`:
   ```bash
   kubectl label node <node-name> pool=gpunode
   ```
5. Deploy the pod using the YAML configuration file in the `k8s` directory:
   ```bash
   kubectl apply -f k8s/psy-model-deployment.yml
   ```

## Automatic Deployment
The repository is configured with **GitHub Actions** for automated deployment. All steps, including Docker build, push, and AKS deployment, are performed automatically upon code changes.