FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
# Use npm install for development (more flexible)
# For production, consider using npm ci with package-lock.json
RUN npm install

# Copy application code
COPY . .

# Build the application (only for production)
# For development, this is skipped as we use npm run dev
# RUN npm run build

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]

