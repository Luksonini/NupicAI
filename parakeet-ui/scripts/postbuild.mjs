import { readdir, readFile, writeFile } from 'node:fs/promises';

const outDir = new URL('../out/', import.meta.url);
const englishPages = [new URL('en.html', outDir)];

async function collectHtml(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const url = new URL(entry.name + (entry.isDirectory() ? '/' : ''), directory);
    if (entry.isDirectory()) await collectHtml(url);
    else if (entry.name.endsWith('.html')) englishPages.push(url);
  }
}

await collectHtml(new URL('en/', outDir));
for (const page of englishPages) {
  const html = await readFile(page, 'utf8');
  await writeFile(page, html.replace('<html lang="pl">', '<html lang="en">'));
}
