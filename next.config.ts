import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // A vestigial package.json sits in the parent directory; pin the root so
  // Turbopack doesn't infer it and warn on every build.
  turbopack: { root: __dirname },
  serverExternalPackages: ["unpdf", "mammoth", "@qdrant/js-client-rest"],
};

export default nextConfig;
