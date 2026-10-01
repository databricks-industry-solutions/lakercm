/* ESLint config for the reviewer frontend.
 *
 * `npm run lint` was in package.json with every plugin installed but NO config
 * file, so it failed on any invocation with "ESLint couldn't find a
 * configuration file" — the script had never been runnable.
 *
 * eslint 8 (the pinned major) reads this legacy .eslintrc format; flat config
 * (eslint.config.js) is eslint 9. `.cjs` because package.json sets
 * "type": "module", which would otherwise make this file ESM.
 */
module.exports = {
  root: true,
  env: { browser: true, es2021: true },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    // The build uses @vitejs/plugin-react's automatic JSX runtime, so JSX needs
    // no React import; without this preset every JSX tag is a react-in-jsx-scope
    // error (659 of them).
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
  ],
  // The lint script globs the whole directory, so the built bundle and the
  // dependencies have to be excluded here or `npm run lint` lints dist/.
  ignorePatterns: ['dist', 'node_modules', '.eslintrc.cjs', 'coverage'],
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module' },
  settings: { react: { version: 'detect' } },
  plugins: ['react-refresh'],
  rules: {
    // This codebase does not use PropTypes anywhere — it is a deliberate style,
    // not 500 missing validations. Leaving the rule on made `npm run lint`
    // useless: the real findings were buried under prop-types noise.
    'react/prop-types': 'off',
    // This codebase deliberately colocates small helper constants and
    // formatters with the component that uses them. The rule is purely an HMR
    // nicety, and satisfying it would mean splitting working files apart.
    'react-refresh/only-export-components': 'off',
  },
  overrides: [
    {
      // AudioWorklet processors run in AudioWorkletGlobalScope, which supplies
      // these globals — they are not browser-window globals.
      files: ['public/*-worklet.js'],
      globals: {
        AudioWorkletProcessor: 'readonly',
        registerProcessor: 'readonly',
        sampleRate: 'readonly',
        currentFrame: 'readonly',
        currentTime: 'readonly',
      },
    },
    {
      // Build/config files run in Node, not the browser, so `process` is defined.
      files: ['vite.config.js', '*.config.js', '*.cjs'],
      env: { node: true, browser: false },
    },
  ],
}
