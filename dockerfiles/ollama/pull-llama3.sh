#!/bin/bash

# Start Ollama server in the background
ollama serve &

# Wait for Ollama server to start
sleep 5

# Pull llama3 model
ollama pull llama3.2:1b

# Wait for the Ollama server to finish 
wait $!