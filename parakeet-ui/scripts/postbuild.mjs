import { readFile, writeFile } from 'node:fs/promises';

const englishPage = new URL('../out/en.html', import.meta.url);
const html = await readFile(englishPage, 'utf8');
await writeFile(englishPage, html.replace('<html lang="pl">', '<html lang="en">'));
