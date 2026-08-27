import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0f1117',
        surface: '#1a1d27',
        surface2: '#22263a',
        border: '#2e3350',
        accent: '#6c8fff',
        'accent-dim': '#3a4e99',
        muted: '#8892a4',
        word: '#fbbf24',
      },
    },
  },
  plugins: [],
};

export default config;
