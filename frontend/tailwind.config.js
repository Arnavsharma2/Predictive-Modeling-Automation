/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        pastel: {
          blue: '#A8D5E2',      // Pastel blue
          powder: '#B0E0E6',    // Powder blue
          baby: '#BFEFFF',      // Baby blue
          mint: '#B5E5CF',      // Mint green
          green: '#C8E6C9',     // Pastel green
          white: '#FAEBD7',     // Antique white
        },
        gray: {
          soft: {
            50: '#F5F5F5',
            100: '#E8E8E8',
            200: '#D3D3D3',
            300: '#C0C0C0',
            400: '#A9A9A9',
            500: '#8B8B8B',
            600: '#6B6B6B',
            700: '#4A4A4A',
          },
        },
        primary: {
          50: '#BFEFFF',  // Baby blue
          100: '#B0E0E6',  // Powder blue
          200: '#A8D5E2', // Pastel blue
          300: '#9BC4CB',
          400: '#8DB4B4',
          500: '#7FA49D',
          600: '#6B8E86',
          700: '#5A7870',
        },
        accent: {
          50: '#C8E6C9',  // Pastel green
          100: '#B5E5CF', // Mint green
          200: '#A3D4B7',
          300: '#91C39F',
          400: '#7FB287',
          500: '#6DA16F',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'slide-in': 'slideIn 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideIn: {
          '0%': { transform: 'translateX(-10px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}

