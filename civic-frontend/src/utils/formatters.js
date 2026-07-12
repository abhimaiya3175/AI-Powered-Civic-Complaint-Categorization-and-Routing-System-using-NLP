import { LANGUAGE_OPTIONS } from './constants';

export const getLanguageLabel = (languageCode) => {
  const option = LANGUAGE_OPTIONS.find((item) => item.value === languageCode);
  return option?.label || (languageCode || 'Unknown');
};

export const formatConfidence = (value) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A';
  }
  return `${(value * 100).toFixed(1)}%`;
};

export const formatDate = (dateString) => {
  if (!dateString) return 'N/A';
  return new Date(dateString).toLocaleString('en-IN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
};
