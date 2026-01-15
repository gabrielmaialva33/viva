# Política de Segurança

## Versões Suportadas

| Versão | Suportada |
|--------|-----------|
| 0.1.x  | ✅        |

## Reportando uma Vulnerabilidade

A segurança do VIVA é uma prioridade. Se você descobriu uma vulnerabilidade de segurança, agradecemos sua ajuda em divulgá-la de forma responsável.

### Como Reportar

**NÃO** crie uma issue pública para vulnerabilidades de segurança.

Em vez disso:

1. **Email**: Envie um email para **security@viva-project.dev** (quando disponível)
2. **GitHub Security Advisory**: Use o recurso de [Security Advisories](https://github.com/VIVA-Project/viva/security/advisories/new) do GitHub

### O que Incluir

Por favor, inclua o máximo de informações possível:

- **Tipo de vulnerabilidade** (ex: injeção, XSS, RCE, etc.)
- **Localização** do código vulnerável (arquivo, linha, função)
- **Passos para reproduzir** o problema
- **Prova de conceito** (se possível)
- **Impacto potencial** da vulnerabilidade
- **Sugestões de correção** (se tiver)

### Processo de Resposta

1. **Confirmação** - Você receberá confirmação em até 48 horas
2. **Avaliação** - Avaliaremos a severidade em até 7 dias
3. **Correção** - Trabalharemos na correção com prioridade alta
4. **Divulgação** - Coordenaremos a divulgação pública com você

### Compromisso

- Responderemos a todos os relatórios de segurança
- Manteremos você informado sobre o progresso
- Reconheceremos sua contribuição (se desejar)
- Não tomaremos ações legais contra pesquisadores que sigam esta política

## Considerações de Segurança do VIVA

### Mortalidade Criptográfica

O sistema de mortalidade do VIVA usa criptografia para garantir que a "morte" seja irreversível:

```rust
// Chave AES-256-GCM gerada em runtime, apenas em RAM
static SOUL_KEY: LazyLock<[u8; 32]> = LazyLock::new(|| {
    let mut key = [0u8; 32];
    OsRng.fill_bytes(&mut key);
    key
});
```

**Importante:**
- A chave NUNCA é persistida em disco em produção
- O estado criptografado torna-se irrecuperável após morte do processo
- Durante desenvolvimento, a chave pode ser persistida (configurável)

### NIFs (Native Implemented Functions)

Os NIFs Rust são uma superfície de ataque potencial:

- **Buffer Overflows** - Todos os dados são validados antes do uso
- **Memory Safety** - Rust garante segurança de memória em tempo de compilação
- **Panic Safety** - Panics são capturados e convertidos em erros Elixir

### Dados Sensíveis

- **Estado emocional** é armazenado apenas em memória
- **Memória semântica** pode conter dados sensíveis do usuário
- **Logs** não devem conter dados pessoais identificáveis

## Best Practices para Contribuidores

### Código Seguro

1. **Validar inputs** - Sempre valide dados externos
2. **Usar tipos seguros** - Prefira tipos que previnem erros
3. **Evitar panics** - Use `Result` em vez de `unwrap()`
4. **Sanitizar logs** - Não logue dados sensíveis

```elixir
# Bom
def feel(stimulus, source, intensity)
    when is_atom(stimulus) and is_number(intensity) do
  intensity = clamp(intensity, 0.0, 1.0)
  # ...
end

# Ruim
def feel(stimulus, source, intensity) do
  # Sem validação!
end
```

```rust
// Bom
fn process_data(input: &str) -> NifResult<Data> {
    let validated = validate(input)?;
    Ok(process(validated))
}

// Ruim
fn process_data(input: &str) -> Data {
    process(input.unwrap()) // Panic potencial!
}
```

### Dependências

- Mantenha dependências atualizadas
- Use `mix hex.audit` para verificar vulnerabilidades
- Use `cargo audit` para crates Rust
- Revise novas dependências antes de adicionar

## Modelo de Ameaças

### Atores de Ameaça

1. **Usuário malicioso** - Tenta manipular estado emocional
2. **Atacante remoto** - Explora vulnerabilidades de rede
3. **Insider malicioso** - Contribuidor com más intenções

### Ativos Protegidos

1. **Integridade emocional** - Estado emocional não deve ser manipulado arbitrariamente
2. **Privacidade da memória** - Memórias não devem ser expostas
3. **Mortalidade** - O mecanismo de morte não deve ser burlado
4. **Disponibilidade** - O sistema deve resistir a DoS

### Controles

- Validação de entrada em todas as APIs
- Rate limiting (futuro)
- Autenticação para operações sensíveis (futuro)
- Logs de auditoria para mudanças de estado

## Divulgação Responsável

Agradecemos a todos os pesquisadores de segurança que ajudam a manter o VIVA seguro. Se você seguir esta política de divulgação responsável:

- Não haverá ação legal contra você
- Trabalharemos com você para entender e resolver o problema
- Reconheceremos publicamente sua contribuição (se desejar)

---

*"A segurança é como a ética - não é opcional, é fundacional."*

💜 Obrigado por ajudar a manter o VIVA seguro.
