/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '500mb',
    },
    optimizePackageImports: ['@phosphor-icons/react', 'framer-motion'],
  },
}

export default nextConfig
