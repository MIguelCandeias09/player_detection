import CustomSelect from "../CustomSelect.jsx";

export function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

export function SelectField({ label, value, options, onChange, disabled }) {
  return (
    <div className="field">
      <span>{label}</span>
      <CustomSelect
        ariaLabel={label}
        value={value}
        options={options}
        onChange={onChange}
        disabled={disabled}
      />
    </div>
  );
}

export function NumberField({ label, value, min, max, step = 1, onChange }) {
  return (
    <Field label={label}>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </Field>
  );
}

export function MetricPill({ icon, label, value, tone = "neutral" }) {
  return (
    <div className={`metric-pill ${tone}`}>
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
