import React from 'react';

export default function PathVisualizer({ steps }) {
  if (!steps) return null;
  return (
    <div className="p-6 bg-white shadow-xl rounded-lg">
      <h2 className="text-xl font-bold mb-4 text-indigo-600">Your Learning Journey</h2>
      <div className="space-y-4">
        {steps.map((s, i) => (
          <div key={i} className="border-l-4 border-indigo-500 pl-4 py-2">
            <p className="font-semibold text-gray-800">{s.title}</p>
            <p className="text-sm text-gray-500">{s.duration}</p>
          </div>
        ))}
      </div>
    </div>
  );
}