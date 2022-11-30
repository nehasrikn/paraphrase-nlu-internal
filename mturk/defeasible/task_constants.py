INSTRUCTIONS="""
<b> Paraphrase the sentence above in three different ways</b> such that its meaning is retained, but the words or sentence structure substantially differ from the original sentence. Make sure that your three paraphrases are <b>different from each other</b> as well.
"""

INPUT_TEMPLATE="""
<crowd-input id='paraphrase_ID' name='paraphrase_ID' spellcheck="true" label='Paraphrase #NUM' required onchange onpropertychange onkeyuponpaste oninput="calculate_BLEU('free_text_id', 'similarity_id', 'original_sentence')"></crowd-input>
<div>Distance from Original Sentence: <span id='similarity_ID'></span></div>
"""

TASK_CONTENT = """
<div id=task1 class='first-container'>
	<div class="left-container">
		<h3>Scenario</h3>
			<crowd-card>
			    <div class="card">
			    	<b><span style='color:#eb8034'>Context</span></b> 
					<br>
					PREMISE
			    </div>
			</crowd-card>
			<br>
			<br>
			<crowd-card>
				<div class="card">
			        <b><span style='color:#7b76db'>Inference</span></b>
					<br>
					HYPOTHESIS
			    </div>
			</crowd-card>
	</div>
	<!-- <span id="vertical-separator"></span> -->
	<div class="right-container">
		<h3><span style='color:#0f6066'>EVIDENCE_TYPE Evidence for Inference</span></h3>
			<crowd-card>
				<div class="card">
					UPDATE
				</div>
			</crowd-card>
		<div style='text-align: left; padding-left: 5px; padding-top: 30px'>
			INSTRUCTIONS							
		</div>
		PARAPHRASES_INPUT
	</div>
</div>
<br>
<div style="position: relative;">
	<crowd-button id="submit_button" form-action="submit">
		Submit
	</crowd-button>
	<div id="errorBox"></div>
	<br>
</div>

"""