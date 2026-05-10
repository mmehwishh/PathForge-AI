const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export async function generateLearningPath(payload) {
  const response = await fetch(`${API_BASE_URL}/api/learning-path/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const error = new Error(data?.detail || 'Failed to generate learning path.');
    error.code = data?.code;
    error.availableTopics = data?.available_topics || [];
    throw error;
  }

  return data;
}
