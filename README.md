# HackMIT 2025 TempoRoll

## Project Overview

TempoRoll is an EEG-based therapeutic music system that analyzes brainwave patterns in real-time to generate or select personalized music for emotional regulation and mental wellness.

## System Architecture

**Data Flow:**
- EEG Device → Frontend → Backend → LLM Analysis → Music Generation/Selection → Audio Playback

**Processing Pipeline:**
- EEG brainwave data → ML emotion classification → LLM therapeutic analysis → Music recommendation → Audio output

## Folder Structure

### `/backend/`
Core server application that processes EEG data and handles music generation/selection:
- Socket server for receiving EEG data from frontend
- ML-based emotion inference from brainwave patterns  
- LLM integration for therapeutic music recommendations
- Music generation via Suno API or selection from existing files
- Audio playback and session management

### `/frontend/`
Nuxt.js web application providing the user interface:
- Dashboard for real-time EEG visualization
- User session management and controls
- WebSocket client for backend communication
- Responsive UI for monitoring brainwave patterns and music playback

### `/ml/`
Machine learning components for emotion classification:
- EEG data preprocessing and feature extraction
- Emotion classification model training and inference
- Model persistence and loading utilities

### `/models/`
Trained machine learning models:
- Brainwave emotion classification models
- Label encoders and preprocessing artifacts

### `/neurosky-comms/`
NeuroSky EEG device communication:
- Device connection and data reading
- Real-time brainwave data streaming
- Socket communication with backend server

### `/suno/`
Music generation API integration:
- Suno API client for generating therapeutic music
- Audio file management and download utilities

## Key Features

- **Real-time EEG Analysis**: Continuous monitoring of brainwave patterns
- **Emotion Classification**: ML-powered emotion inference with 85% accuracy
- **Therapeutic Music**: LLM-generated personalized music recommendations
- **Dual Mode Operation**: Real-time music generation or selection from existing library
- **Session Management**: User profile and session data persistence
