import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        bg: '#111310',
        surface: '#191c18',
        surface2: '#222620',
        border: '#343a32',
        accent: '#45b88a',
        'accent-dim': '#23684f',
        muted: '#939c91',
        word: '#fbbf24',
      },
    },
  },
  plugins: [],
};

export default config;
