#!/usr/bin/env node
import { main } from '../dist/cli.js';

main().catch((err) => {
  // Usage errors reach here; a stack trace would bury the one line that matters.
  console.error(err?.message ? `Error: ${err.message}` : err);
  if (process.env.DSH_DEBUG && err?.stack) console.error(err.stack);
  process.exit(1);
});
