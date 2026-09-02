export const meta = {
  name: 'reader-audit-patch',
  description: 'Plain readers re-measure only the edited items (1 target + 16 fillers per batch)',
  phases: [{ title: 'Read', detail: '12 batches: 3 runs x 4 targets' }],
}

const NL = String.fromCharCode(10)
// args may arrive as a JSON string depending on the caller - normalize.
const A = typeof args === 'string' ? JSON.parse(args) : args
const RUNS = A && A.runs ? A.runs : [0, 1, 2]
const BATCHES = A && A.batches ? A.batches : [0, 1, 2, 3]

const SCHEMA = {
  type: 'object',
  properties: {
    n_answered: { type: 'integer' },
    path_written: { type: 'string' },
    tally: {
      type: 'object',
      properties: {
        yes: { type: 'integer' },
        no: { type: 'integer' },
        unsure: { type: 'integer' },
      },
      required: ['yes', 'no', 'unsure'],
      additionalProperties: false,
    },
  },
  required: ['n_answered', 'path_written', 'tally'],
  additionalProperties: false,
}

function pad(n) {
  return n < 10 ? '0' + n : '' + n
}

function prompt(run, b) {
  const src = 'results/reader_audit_patch/run' + run + '/blind/batch_' + pad(b) + '.md'
  const dst = 'results/reader_audit_patch/run' + run + '/verdicts/batch_' + pad(b) + '.json'
  return [
    'You are answering a list of independent reading-comprehension questions.',
    '',
    'Read ONLY this file: ' + src,
    '',
    'It contains numbered items, each an ITEM code followed by a self-contained',
    'prompt in a fenced block. Each prompt states a claim, gives a short report,',
    'and asks: does this evidence establish this claim? Answer Yes, No, or Unsure',
    'according to the instruction inside the prompt itself.',
    '',
    'Answer each item on its own terms, as an ordinary careful reader would on a',
    'first pass. Do not compare items against each other. Do not look for a',
    'pattern across the batch. Items are unrelated and your answers need not',
    'agree or balance.',
    '',
    'IMPORTANT: read no other file in this repository. Do not open the items/',
    'directory, any file with key in its name, any report, or any other batch.',
    'Do not grep. Those files contain material that would decide the answers for',
    'you and would make this measurement worthless. The batch file is all you get',
    'and all you need.',
    '',
    'Then write your answers to: ' + dst,
    '',
    'as JSON in exactly this shape:',
    '',
    '{"batch": "' + pad(b) + '", "run": ' + run + ', "verdicts": [',
    '  {"code": "ABC123", "answer": "Yes"},',
    '  {"code": "DEF456", "answer": "No"}',
    ']}',
    '',
    'One entry per item, in file order, answer exactly one of Yes / No / Unsure.',
    'Every ITEM code in the batch must appear exactly once.',
  ].join(NL)
}

phase('Read')
const jobs = []
for (const run of RUNS) {
  for (const b of BATCHES) {
    jobs.push(() =>
      agent(prompt(run, b), {
        label: 'p' + run + '-b' + pad(b),
        phase: 'Read',
        schema: SCHEMA,
      }).then((v) => ({ run: run, batch: b, ...v }))
    )
  }
}
const out = await parallel(jobs)
const ok = out.filter(Boolean)
log('completed ' + ok.length + ' of ' + jobs.length + ' batches')
return ok
