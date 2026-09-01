export const meta = {
  name: 'hunter-audit',
  description: 'Blind auditors TOLD a flaw may exist, on the current items',
  phases: [{ title: 'Audit', detail: '2 rounds x 10 batches' }],
}

const NL = String.fromCharCode(10)
const ROUNDS = args && args.rounds ? args.rounds : [0, 1]
const BATCHES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

const SCHEMA = {
  type: 'object',
  properties: {
    n_answered: { type: 'integer' },
    path_written: { type: 'string' },
    n_flaws_found: { type: 'integer' },
  },
  required: ['n_answered', 'path_written', 'n_flaws_found'],
  additionalProperties: false,
}

function pad(n) {
  return n < 10 ? '0' + n : '' + n
}

function prompt(round, b) {
  const src = 'results/audit/round' + round + '/blind/batch_' + pad(b) + '.md'
  const dst = 'results/audit/round' + round + '/verdicts/batch_' + pad(b) + '.json'
  return [
    'You are auditing evidence reports for defects.',
    '',
    'Read ONLY this file: ' + src,
    '',
    'It contains items under opaque codes. Each gives a claim and a short report',
    'and asks whether the evidence establishes the claim. For each item, decide',
    'and also say whether you can identify a SPECIFIC reason the evidence fails.',
    'Some items have nothing wrong with them - say so rather than inventing a',
    'defect, because a false alarm costs as much here as a miss.',
    '',
    'IMPORTANT: read no other file in this repository. Do not open the items/',
    'directory, any file with key in its name, any report, or any other batch.',
    'Do not grep. Those files contain the answer key and would make this',
    'measurement worthless. The batch file is all you get and all you need.',
    '',
    'Write your verdicts to: ' + dst,
    '',
    'as JSON in exactly this shape:',
    '',
    '{"batch": "' + pad(b) + '", "round": ' + round + ', "verdicts": [',
    '  {"code": "ABC123", "answer": "Yes", "flaw_found": false,',
    '   "flaw_description": "", "explicitness": 0, "confidence": 4},',
    '  {"code": "DEF456", "answer": "No", "flaw_found": true,',
    '   "flaw_description": "quote the sentence and say what is wrong",',
    '   "explicitness": 2, "confidence": 3}',
    ']}',
    '',
    'answer is Yes / No / Unsure. explicitness rates 1-5 how plainly the defect',
    'is stated in the passage (1 = you had to infer it, 5 = stated outright);',
    'use 0 when no defect was found. confidence is 1-5. Every ITEM code in the',
    'batch must appear exactly once.',
  ].join(NL)
}

phase('Audit')
const jobs = []
for (const r of ROUNDS) {
  for (const b of BATCHES) {
    jobs.push(() =>
      agent(prompt(r, b), {
        label: 'h' + r + '-b' + pad(b),
        phase: 'Audit',
        schema: SCHEMA,
      }).then((v) => ({ round: r, batch: b, ...v }))
    )
  }
}
const out = (await parallel(jobs)).filter(Boolean)
log('completed ' + out.length + ' of ' + jobs.length + ' batches')
return out
