const fs = require('fs');
const Ajv = require('ajv');

const ajv = new Ajv({ allErrors: true });
const ritualSchema = JSON.parse(fs.readFileSync('./schemas/ritual.schema.json'));

// Generate IR for a ritual
function generateIR(ritual) {
  return {
    id: ritual.ritual,
    agent: ritual.agent,
    preconditions: ritual.requires,
    defenses: ritual.defense.map(d => ({
      type: d.type,
      logic: d.logic,
      params: d.parameters || {}
    })),
    responses: ritual.response,
    recursive: ritual.recursive,
    awareness: {
      reflection: ritual.awareness.self_reflection,
      state: ritual.awareness.state
    },
    audit: {
      origin: ritual.audit.origin,
      timestamp: ritual.audit.timestamp,
      verifiedBy: ritual.audit.verified_by,
      signalChain: ritual.audit.signal_chain || []
    }
  };
}

// Generate executable graph
function generateGraph(ir) {
  return {
    nodes: [
      { id: 'start', type: 'start', data: { agent: ir.agent } },
      ...ir.preconditions.map((p, i) => ({
        id: `precondition_${i}`,
        type: 'check',
        data: { condition: p }
      })),
      ...ir.defenses.map((d, i) => ({
        id: `defense_${i}`,
        type: d.type,
        data: { logic: d.logic, params: d.params }
      })),
      ...ir.responses.map((r, i) => ({
        id: `response_${i}`,
        type: 'action',
        data: { action: r }
      })),
      ...(ir.recursive.enabled ? [{
        id: 'recursive',
        type: 'loop',
        data: {
          target: ir.recursive.target_ritual,
          maxDepth: ir.recursive.max_depth,
          condition: ir.recursive.condition
        }
      }] : []),
      { id: 'audit', type: 'log', data: ir.audit }
    ],
    edges: [
      ...ir.preconditions.map((_, i) => ({
        source: i === 0 ? 'start' : `precondition_${i-1}`,
        target: `precondition_${i}`
      })),
      ...ir.defenses.map((_, i) => ({
        source: i === 0 ? `precondition_${ir.preconditions.length-1}` : `defense_${i-1}`,
        target: `defense_${i}`
      })),
      ...ir.responses.map((_, i) => ({
        source: i === 0 ? `defense_${ir.defenses.length-1}` : `response_${i-1}`,
        target: `response_${i}`
      })),
      ...(ir.recursive.enabled ? [{
        source: `response_${ir.responses.length-1}`,
        target: 'recursive'
      }, { source: 'recursive', target: 'start' }] : []),
      {
        source: ir.recursive.enabled ? 'recursive' : `response_${ir.responses.length-1}`,
        target: 'audit'
      }
    ]
  };
}

// Load and validate ritual
function compileRitual(filePath) {
  const ritual = JSON.parse(fs.readFileSync(filePath));
  const validate = ajv.compile(ritualSchema);
  if (!validate(ritual)) {
    console.error("Validation errors:", validate.errors);
    process.exit(1);
  }

  const ir = generateIR(ritual);
  const graph = generateGraph(ir);

  console.log("✅ Ritual Valid!");
  console.log("📜 Intermediate Representation:", JSON.stringify(ir, null, 2));
  console.log("🌐 Executable Graph:", JSON.stringify(graph, null, 2));

  return { ir, graph, ritual };
}

// Example compilation
compileRitual('./rituals/neural_signal_initiation.json');