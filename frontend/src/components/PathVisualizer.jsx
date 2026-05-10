import React from 'react';

export default function PathVisualizer({ learningPath }) {
  if (!learningPath) {
    return (
      <section className="empty-state">
        <span>Roadmap preview</span>
        <p>Your generated path will appear here after the API responds.</p>
      </section>
    );
  }

  return (
    <section className="roadmap-card">
      <div className="roadmap-header">
        <div>
          <span className="eyebrow">Generated roadmap</span>
          <h2>{learningPath.topic}</h2>
        </div>
        <div className="summary-pills">
          <span>{learningPath.level}</span>
          <span>{learningPath.weekly_hours} hrs/week</span>
          <span>{learningPath.total_weeks} weeks</span>
        </div>
      </div>

      <div className="phase-strip">
        {learningPath.phases.map((phase) => (
          <span key={phase}>{phase}</span>
        ))}
      </div>

      {learningPath.courses.length < 3 && (
        <div className="limited-roadmap-note">
          This topic has limited data, so we are showing the courses currently available.
        </div>
      )}

      <div className="timeline">
        {learningPath.courses.map((course, index) => (
          <article className="timeline-item" key={`${course.title}-${index}`}>
            <div className="week-marker">
              <strong>
                {course.start_week === course.end_week
                  ? `Week ${course.start_week}`
                  : `Weeks ${course.start_week}-${course.end_week}`}
              </strong>
              <span>{course.phase}</span>
            </div>

            <div className="course-card">
              <h3>{course.title}</h3>
              <p>{course.description}</p>
              <div className="course-meta">
                <span>{course.estimated_hours} total hours</span>
                <span>{course.estimated_weeks} week plan</span>
                <span>{course.status}</span>
              </div>
            </div>
          </article>
        ))}
      </div>

      <div className="total-hours">
        Total estimated study time: {learningPath.total_estimated_hours} hours
      </div>
    </section>
  );
}
