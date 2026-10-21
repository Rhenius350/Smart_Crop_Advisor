import axios from 'axios';

// FastAPI Backend Endpoint (localhost:8000)
const API_BASE_URL = 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000, // 15 second timeout
});

// User-friendly error message formatter
const handleApiError = (error) => {
  if (error.response) {
    // Server responded with error status
    const message = error.response.data?.detail || error.response.data?.message || 'Server error occurred.';
    return new Error(message);
  } else if (error.request) {
    // Request made but no response received (Backend down or CORS issue)
    return new Error('Unable to connect to the prediction server. Please ensure the FastAPI backend is running on http://localhost:8000.');
  } else {
    return new Error(error.message || 'An unexpected request error occurred.');
  }
};

export const predictCrop = async (inputData) => {
  try {
    const response = await apiClient.post('/predict', inputData);
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

export const explainPrediction = async (inputData) => {
  try {
    const response = await apiClient.post('/explain', inputData);
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

export const getModelInfo = async () => {
  try {
    const response = await apiClient.get('/model-info');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

export const getPerformance = async () => {
  try {
    const response = await apiClient.get('/performance');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

export const getFeatureImportance = async () => {
  try {
    const response = await apiClient.get('/feature-importance');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

export const getResearchData = async () => {
  try {
    const response = await apiClient.get('/research');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

export const getHealth = async () => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

export default apiClient;
