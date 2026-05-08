import tf from '@tensorflow/tfjs-node';

async function traindModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Primeira camada de rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)

    //80 neurônios = aqui coloquei tudo isso porque tem pouca base de treino
    // quanto mais neurônios, mais complexo a rede pode entender e consequentemente, amsi processamento ela vai usar

    //A ReLu age como um filtro:
    // É como se ela deixasse somente os dados intessantes seguirem viagem na rede
    // Se a indformação chegou nesse neurônio é positiva, passa pra frente!
    // se for zero ou negativa, pode jogar fora, não vai servir para nada

    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Saída: 3 neurônios
    // um para cada categoria (premium, medium, basic)

    // activation: softmax normaliza a saída em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // Compilando o documento
    // optimizar Asam (Adaptative Moment Estimation)
    // é um treinador pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com histórico de erros e acertos

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria) com a resposta certa
    // a categoria premium será sempre [1, 0, 0]

    // Quanto mais distante da previsão do modelo da resposta correta maior o erro (loss)
    // Exemplo clássico: classificação de imagens, recomendação, categorização de usuário
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // Treinando o modelo
    // verbose: desabilita o log interno (e usa só callback)
    // epochs: quantidade de vezes que vai rodar no dataset
    // shuffle: embaralha os dados, para evitar bias
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                // onEpochEnd: (epoch, logs) => console.log(
                //     `Epoch ${epoch}: loss = ${logs.loss}`
                // )
            }
        }
    )

    return model;
}

async function predict(model, pessoaTensorNormalizado) {
    // Transformar o array js para tensor (tfjs)
    const tfInput = tf.tensor2d(pessoaTensorNormalizado)

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// Quanto mais dados melhor!
// assim o algoritmo consegue entender melhor os padrões complexos dos dados

const model = await traindModel(inputXs, outputYs);

const pessoa = { nome: "Zé", idade: 28, cor: "verde", localização: "Curitiba" }
// Normalizando a idade da nova pessoa usando o mesmo padrão de treino
// Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2

const pessoaTensorNormalizado =[
    [
        0.2, // idade normalizada
        1,   // azul
        0,   // vermelho
        0,   // verde
        1,   // São Paulo
        0,   // Rio
        0    // Curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado);
const results = predictions
    .sort((a, b) => b.prob - a.prob) // Ordena por probabilidade decrescente
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`) // Mapeia para nome do label e formata a probabilidade
    .join('\n')

console.log(results);
