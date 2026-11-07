// Captura una respuesta real de POST /predict desde Swagger UI.
import { createRequire } from 'node:module';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [baseUrl = 'http://127.0.0.1:8000', output = 'evidence/container/api-container.png'] = process.argv.slice(2);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const body = await fs.readFile(path.join(root, 'examples/p3-request.json'), 'utf8');
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
  await page.goto(`${baseUrl}/docs`, { waitUntil: 'networkidle' });
  const operation = page.locator('.opblock-post').filter({ hasText: '/predict' });
  await operation.locator('.opblock-summary').click();
  await operation.getByRole('button', { name: 'Try it out' }).click();
  await operation.locator('textarea').fill(body);
  const responsePromise = page.waitForResponse(response =>
    response.url() === `${baseUrl}/predict` && response.request().method() === 'POST');
  await operation.getByRole('button', { name: 'Execute', exact: true }).click();
  const response = await responsePromise;
  if (response.status() !== 200) throw new Error(`HTTP ${response.status()}`);
  await operation.locator('.responses-inner .response-col_description').filter({ hasText: 'predictions' }).first().waitFor();
  await fs.mkdir(path.dirname(output), { recursive: true });
  await page.screenshot({ path: output, fullPage: true });
  console.log(`Captura de la petición real guardada en ${output}`);
} finally {
  await browser.close();
}
