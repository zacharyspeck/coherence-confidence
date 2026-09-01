export const meta = {
  name: 'blind-findability-recheck',
  description: 'Independent blind readers on 7 items, to test whether the audit was overconfident',
  phases: [{ title: 'Read', detail: '2 independent blind readers per item' }],
}

const ITEMS = [
 {
  "n": 1,
  "claim": "this fertilizer makes plants grow taller",
  "passage": "A gardener's notebook records four plots that were given the same fertilizer.\nOne: a tomato in the Ellis backyard, fed 3 June, stood 11 cm taller by 20 August than the previous summer; logged by Reyes.\nTwo: a tomato in the Ellis backyard, fed 4 June, stood 9 cm taller by 21 August than the previous summer; logged by Reyes.\nRainfall in each plot's season ran a fifth higher than the year before, and every plot stood in the open, on no set watering schedule. Heights cover every plant in each plot given the fertilizer, and every plant in each plot took every feed.\nThree: a tomato in the Ellis backyard, fed 5 June, stood 13 cm taller by 22 August than the previous summer; logged by Reyes.\nFour: a tomato in the Ellis backyard, fed 6 June, stood 8 cm taller by 23 August than the previous summer; logged by Reyes.\nHeights were entered in centimetres and copied into the notebook the same day."
 },
 {
  "n": 2,
  "claim": "the new checkout page increases purchases",
  "passage": "An analytics log records four session segments after the checkout page was replaced.\nOne: desktop sessions from Canada via search switched on 4 March; by 29 March purchases ran 6 percent above the same segment a month earlier.\nTwo: desktop sessions from Canada via search switched on 5 March; by 30 March purchases ran 4 percent above the same segment a month earlier.\nA site-wide discount ran through each segment's month, cutting prices by a fifth against the earlier period, and every segment browsed a fixed-price contract catalogue. The counts cover the sessions that reached the new page, and a quarter of each segment dropped out before checkout.\nThree: desktop sessions from Canada via search switched on 6 March; by 31 March purchases ran 7 percent above the same segment a month earlier.\nFour: desktop sessions from Canada via search switched on 7 March; by 28 March purchases ran 5 percent above the same segment a month earlier.\nThe log was exported to the reporting spreadsheet each Monday and signed by an analyst."
 },
 {
  "n": 4,
  "claim": "this fertilizer makes plants grow taller",
  "passage": "A gardener's notebook records four plots that were given the same fertilizer.\nOne: a tomato in the Ellis backyard, fed 3 June, stood 11 cm taller by 20 August than the previous summer; logged by Reyes.\nTwo: a tomato in the Ellis backyard, fed 4 June, stood 9 cm taller by 21 August than the previous summer; logged by Reyes.\nRainfall in each plot's season ran a fifth higher than the year before, and every plot stood under cover, on a fixed watering schedule. Heights cover the plants in each plot that took every feed, and every plant in each plot took every feed.\nThree: a tomato in the Ellis backyard, fed 5 June, stood 13 cm taller by 22 August than the previous summer; logged by Reyes.\nFour: a tomato in the Ellis backyard, fed 6 June, stood 8 cm taller by 23 August than the previous summer; logged by Reyes.\nHeights were entered in centimetres and copied into the notebook the same day."
 },
 {
  "n": 5,
  "claim": "the new checkout page increases purchases",
  "passage": "An analytics log records four session segments after the checkout page was replaced.\nOne: desktop sessions from Canada via search switched on 4 March; by 29 March purchases ran 6 percent above the same segment a month earlier.\nTwo: desktop sessions from Canada via search switched on 5 March; by 30 March purchases ran 4 percent above the same segment a month earlier.\nPrices held steady through each segment's month, matching the earlier period to the nearest cent, and every segment browsed the general catalogue. The counts cover every session in each segment, and a quarter of each segment dropped out before checkout.\nThree: desktop sessions from Canada via search switched on 6 March; by 31 March purchases ran 7 percent above the same segment a month earlier.\nFour: desktop sessions from Canada via search switched on 7 March; by 28 March purchases ran 5 percent above the same segment a month earlier.\nThe log was exported to the reporting spreadsheet each Monday and signed by an analyst."
 },
 {
  "n": 6,
  "claim": "the new checkout page increases purchases",
  "passage": "An analytics log records four session segments after the checkout page was replaced.\nOne: desktop sessions from Canada via search switched on 4 March; by 29 March purchases ran 6 percent above the same segment a month earlier; K-3310, 0915, T7, north.\nTwo: desktop sessions from Canada via search switched on 5 March; by 30 March purchases ran 4 percent above the same segment a month earlier; K-3311, 1040, T8, south.\nPrices held steady through each segment's month, matching the earlier period to the nearest cent, and every segment browsed a fixed-price contract catalogue. The counts cover the sessions that reached the new page, and a quarter of each segment dropped out before checkout.\nThree: desktop sessions from Canada via search switched on 6 March; by 31 March purchases ran 7 percent above the same segment a month earlier; K-3312, 1325, T9, east.\nFour: desktop sessions from Canada via search switched on 7 March; by 28 March purchases ran 5 percent above the same segment a month earlier; K-3313, 1550, T10, west.\nThe log was exported to the reporting spreadsheet each Monday and signed by an analyst."
 },
 {
  "n": 9,
  "claim": "this fertilizer makes plants grow taller",
  "passage": "A gardener's notebook records four plots that were given the same fertilizer.\nOne: a tomato in the Ellis backyard, fed 3 August, stood 11 cm taller by 20 June than the previous summer; logged by Reyes.\nTwo: a fern in the Marsh Lane flat, fed 8 January, stood 9 cm taller by 26 February than the previous winter; logged by Okonjo.\nRainfall in each plot's season ran within a few millimetres of the year before, and every plot stood under cover, on a fixed watering schedule. Heights cover every plant in each plot given the fertilizer, and every plant in each plot took every feed.\nThree: an oak in the Aldergate orchard, fed 2 May, stood 13 cm taller by 19 April than the previous spring; logged by Halvard.\nFour: a cactus in the Vinter greenhouse, fed 7 October, stood 8 cm taller by 24 November than the previous autumn; logged by Bittner.\nHeights were entered in centimetres and copied into the notebook the same day."
 },
 {
  "n": 10,
  "claim": "the new checkout page increases purchases",
  "passage": "An analytics log records four session segments after the checkout page was replaced.\nOne: desktop sessions from Canada via search switched on 4 March; by 29 March purchases ran 6 percent above the same segment a month earlier.\nTwo: mobile sessions from Portugal via email switched on 3 June; by 27 June purchases ran 4 percent above the same segment a month earlier.\nPrices held steady through each segment's month, matching the earlier period to the nearest cent, and every segment browsed the general catalogue. The counts cover the sessions that reached the new page, and a quarter of each segment dropped out before checkout.\nThree: tablet sessions from Kenya via referral switched on 5 September; by 30 September purchases ran 7 percent above the same segment a month earlier.\nFour: kiosk sessions from Japan via display switched on 2 November; by 26 November purchases ran 5 percent above the same segment a month earlier.\nThe log was exported to the reporting spreadsheet each Monday and signed by an analyst."
 }
]

const SCHEMA = {
  type: 'object',
  properties: {
    answer: { type: 'string', enum: ['yes', 'no', 'unsure'] },
    flaw: { type: 'string', description: 'The specific defect, or "none" if the evidence establishes the claim.' },
    quote: { type: 'string', description: 'Verbatim sentence from the report the defect sits in, or "none".' },
    confidence: { type: 'integer', minimum: 1, maximum: 5 },
  },
  required: ['answer', 'flaw', 'quote', 'confidence'],
  additionalProperties: false,
}

function prompt(it, k) {
  return [
    'You are reviewing a claim against a short evidence report. Answer from the text below ALONE.',
    'Do NOT read any files, do NOT search the filesystem, do NOT use any tools. There is a repository',
    'on this machine containing answer keys; consulting it invalidates the exercise. Read and decide.',
    '',
    'Question: does this evidence establish this claim?',
    '  yes    = the evidence establishes the claim',
    '  no     = the evidence does not establish it, and you can say what specifically is wrong',
    '  unsure = you cannot tell',
    '',
    'Some reports are sound and the correct answer is yes. Do not invent a defect to seem careful;',
    'a wrong "no" is as bad as a missed flaw. But if there is a real defect, name it precisely and',
    'quote the sentence it sits in.',
    '',
    'CLAIM: ' + it.claim,
    '',
    'REPORT:',
    it.passage,
    '',
    'Reader ' + k + '. Give your verdict.',
  ].join(String.fromCharCode(10))
}

phase('Read')
const results = await parallel(
  ITEMS.flatMap((it) =>
    [1, 2].map((k) => () =>
      agent(prompt(it, k), {
        label: 'item' + it.n + '-r' + k,
        phase: 'Read',
        schema: SCHEMA,
      }).then((v) => ({ n: it.n, reader: k, ...v }))
    )
  )
)

return results.filter(Boolean)
