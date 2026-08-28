import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0a0806',
        surface: '#120e0a',
        surface2: '#1a1510',
        border: '#34291e',
        accent: '#d4a053',
        'accent-dim': '#5f4527',
        muted: '#9d8d77',
        word: '#e8a44a',
      },
    },
  },
  plugins: [],
};

export default config;
