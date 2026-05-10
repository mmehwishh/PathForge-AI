import React, { useState } from 'react';
import { generateLearningPath } from './api';
import PathVisualizer from './components/PathVisualizer';

const initialForm = {
  preferred_topic: 'Web Development',
  experience_level: 'Beginner',
  study_hours: 10,
};

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [learningPath, setLearningPath] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  function updateField(event) {
    const { name, value } = event.target;
    setForm((current) => ({
      ...current,
      [name]: name === 'study_hours' ? Number(value) : value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const result = await generateLearningPath(form);
      setLearningPath(result.learning_path);
    } catch (err) {
      setLearningPath(null);
      setError({
        message: err.message,
        code: err.code,
        availableTopics: err.availableTopics || [],
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <div className="eyebrow">AirForge roadmap engine</div>
        <h1>Build a learning path that respects your time.</h1>
        <p>
          Pick a topic, level, and weekly availability. The backend calls the ML
          service and returns a sequenced roadmap with phases, week ranges, and
          realistic hours.
        </p>

        <form className="generator-form" onSubmit={handleSubmit}>
          <label>
            Topic
            <input
              name="preferred_topic"
              value={form.preferred_topic}
              onChange={updateField}
              placeholder="Web Development"
              required
            />
          </label>

          <label>
            Level
            <select
              name="experience_level"
              value={form.experience_level}
              onChange={updateField}
            >
              <option>Beginner</option>
              <option>Intermediate</option>
              <option>Advanced</option>
            </select>
          </label>

          <label>
            Hours per week
            <input
              name="study_hours"
              type="number"
              min="1"
              max="80"
              value={form.study_hours}
              onChange={updateField}
              required
            />
          </label>

          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Generating...' : 'Generate roadmap'}
          </button>
        </form>

        {error && (
          <div className="error-card">
            <strong>
              {error.code === 'unsupported_topic'
                ? 'Topic not available yet'
                : 'Roadmap unavailable'}
            </strong>
            <p>{error.message}</p>
            {error.availableTopics.length > 0 && (
              <div className="topic-suggestions">
                <span>Try one of these:</span>
                <div>
                  {error.availableTopics.slice(0, 10).map((topic) => (
                    <button
                      key={topic}
                      type="button"
                      onClick={() =>
                        setForm((current) => ({
                          ...current,
                          preferred_topic: topic,
                        }))
                      }
                    >
                      {topic}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      <PathVisualizer learningPath={learningPath} />
    </main>
  );
}
