# Sistema de Perguntas e Respostas com RAG (Retrieval-Augmented Generation)

Projeto de Inteligência Artificial focado na implementação de um sistema de perguntas e respostas utilizando **recuperação semântica (RAG)** com embeddings e indexação vetorial via **FAISS**.

Inspirado nas aulas do Curso de Extensão “Fundamentos e Operação Prática em IA” do Senai Cimatec, O sistema permite consultar documentos locais e gerar respostas baseadas em contexto relevante, reduzindo respostas genéricas e aumentando a precisão.

---

## 📝 Licença

### Este projeto é fornecido como está para fins educacionais e de demonstração.

- Desenvolvido por: Marcos Menezes
- Data de Criação: Março de 2026
- Data de Postagem no GitHub: Abril 2026
  
---

## 🎯 Objetivo

Simular um sistema de IA aplicado a cenários reais, onde é necessário responder perguntas com base em uma base de conhecimento, como:

- documentação técnica  
- manuais  
- bases internas de empresas  
- conteúdos educacionais  

---

## ⚙️ Tecnologias utilizadas

- Python  
- LangChain  
- FAISS (Facebook AI Similarity Search)  
- Hugging Face Embeddings (`sentence-transformers`)  
- Streamlit  

---

## 🧠 Como funciona o sistema

O pipeline segue as etapas clássicas de um sistema RAG:

1. **Ingestão de documentos**  
   Arquivos `.txt` são carregados como base de conhecimento.

2. **Chunking**  
   Os textos são divididos em partes menores para melhorar a precisão da busca.

3. **Geração de embeddings**  
   Cada trecho é transformado em um vetor numérico que representa seu significado semântico.

4. **Indexação vetorial (FAISS)**  
   Os embeddings são armazenados em uma estrutura otimizada para busca por similaridade.

5. **Processamento da pergunta**  
   A pergunta do usuário também é convertida em embedding.

6. **Recuperação semântica**  
   O sistema busca os trechos mais relevantes com base na similaridade entre vetores.

7. **Construção da resposta**  
   A resposta é gerada utilizando os trechos recuperados como contexto.

---


## 📊 Diferenciais do projeto

- Implementação completa de pipeline RAG  
- Uso de embeddings semânticos modernos  
- Indexação eficiente com FAISS  
- Recuperação baseada em similaridade vetorial  
- Interface interativa com Streamlit  
- Exibição de fontes utilizadas na resposta  

---

## ⚠️ Limitações

- Base de documentos limitada  
- Não utiliza modelo generativo avançado (LLM completo)  
- Qualidade da resposta depende diretamente da qualidade dos documentos  

---

## 🚀 Como executar

```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
