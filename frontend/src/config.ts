/**
 * Frontend configuration using environment variables
 */

// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const API_VERSION = import.meta.env.VITE_API_VERSION || 'v1';
export const API_URL = `${API_BASE_URL}/api/${API_VERSION}`;

// Development Configuration
export const DEV_MODE = import.meta.env.VITE_DEV_MODE === 'true' || import.meta.env.DEV;
export const ENABLE_LOGGING = import.meta.env.VITE_ENABLE_LOGGING === 'true' || DEV_MODE;

// Search Configuration
export const DEFAULT_SEARCH_TYPE = import.meta.env.VITE_DEFAULT_SEARCH_TYPE || 'hybrid';
export const MAX_SEARCH_RESULTS = parseInt(import.meta.env.VITE_MAX_SEARCH_RESULTS || '10');
export const MIN_SEARCH_SCORE = parseFloat(import.meta.env.VITE_MIN_SEARCH_SCORE || '0.0');

// Upload Configuration
export const MAX_FILE_SIZE = parseInt(import.meta.env.VITE_MAX_FILE_SIZE || '52428800'); // 50MB
export const SUPPORTED_FILE_TYPES = (import.meta.env.VITE_SUPPORTED_FILE_TYPES || 'pdf,docx,md,txt').split(',');

// UI Configuration
export const ENABLE_DARK_MODE = import.meta.env.VITE_ENABLE_DARK_MODE === 'true';
export const ENABLE_ANIMATIONS = import.meta.env.VITE_ENABLE_ANIMATIONS === 'true';
export const RESULTS_PER_PAGE = parseInt(import.meta.env.VITE_RESULTS_PER_PAGE || '10');

// Logging utility
export const log = (...args: any[]) => {
  if (ENABLE_LOGGING) {
    console.log('[RAG Platform]', ...args);
  }
};

export const logError = (...args: any[]) => {
  if (ENABLE_LOGGING) {
    console.error('[RAG Platform Error]', ...args);
  }
};

// Environment validation
if (!API_BASE_URL) {
  throw new Error('VITE_API_BASE_URL is required');
}

export const config = {
  api: {
    baseUrl: API_BASE_URL,
    version: API_VERSION,
    url: API_URL,
  },
  dev: {
    mode: DEV_MODE,
    logging: ENABLE_LOGGING,
  },
  search: {
    defaultType: DEFAULT_SEARCH_TYPE,
    maxResults: MAX_SEARCH_RESULTS,
    minScore: MIN_SEARCH_SCORE,
  },
  upload: {
    maxFileSize: MAX_FILE_SIZE,
    supportedTypes: SUPPORTED_FILE_TYPES,
  },
  ui: {
    darkMode: ENABLE_DARK_MODE,
    animations: ENABLE_ANIMATIONS,
    resultsPerPage: RESULTS_PER_PAGE,
  },
};